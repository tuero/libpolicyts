// File: tree_viz.cpp
// Description: Tree visualizer

#include <libpolicyts/tree_viz.h>

#include "tree_layout.h"

#define GL_SILENCE_DEPRECATION
#include <GLFW/glfw3.h>
#include <imgui.h>
#include <imgui_impl_glfw.h>
#include <imgui_impl_opengl3.h>
#include <imgui_internal.h>

#include <spdlog/spdlog.h>

#include <algorithm>
#include <cmath>
#include <print>

namespace libpts::treeviz {

namespace {
void glfw_error_callback(int error, const char *description)
{
    std::print("GLFW Error {:d}: {:s}\n", error, description);
}

struct Color {
    float r;
    float g;
    float b;
    float a;
};

constexpr Color CLEAR_COLOR{.r = 0.1f, .g = 0.1f, .b = 0.12f, .a = 1.0f};

// Internal renderable node with layout positions
struct VisualNode {
    int id;
    std::optional<int> parent_id;
    int action_taken;
    std::string label;
    std::size_t source_index;
    float x = 0.0f;
    float y = 0.0f;
};

struct Edge {
    std::size_t to;
    std::size_t from;
    int action_taken;
};

struct VisualTree {
    std::vector<VisualNode> nodes;
    std::unordered_map<int, std::size_t> index_by_id;
    std::vector<Edge> edges;    // (from, to, action)
};

// Cached visual tree to prevent recomputing every frame
struct VisualTreeCache {
    uint64_t structure_fingerprint = 0;
    bool valid = false;

    VisualTree tree;

    void clear()
    {
        structure_fingerprint = 0;
        valid = false;
        tree = {};
    }
};

struct ImageTexture {
    GLuint id = 0;
    int width = 0;
    int height = 0;
};

auto compute_prepared_tree_fingerprint(const PreparedTree &prepared) -> uint64_t
{
    uint64_t fp = 0xcbf29ce484222325ull;    // NOLINT(*-magic-numbers)
    detail::hash_combine(fp, static_cast<std::uint64_t>(prepared.nodes.size()));

    for (const auto &n : prepared.nodes) {
        detail::hash_combine(fp, static_cast<std::uint64_t>(n.id));
        detail::hash_combine(fp, n.parent_id ? static_cast<std::uint64_t>(*n.parent_id) : 0);
    }

    return fp;
}

// GL key checks
auto IsKeyDown(GLFWwindow *window, int key) -> bool
{
    return glfwGetKey(window, key) == GLFW_PRESS;
}

auto IsSpaceDown(GLFWwindow *window) -> bool
{
    return IsKeyDown(window, GLFW_KEY_SPACE);
}

[[maybe_unused]] auto IsCommandDown(GLFWwindow *window) -> bool
{
    return IsKeyDown(window, GLFW_KEY_LEFT_SUPER) || IsKeyDown(window, GLFW_KEY_RIGHT_SUPER);
}

auto WantPanDrag(GLFWwindow *window) -> bool
{
    // Standard mouse users
    if (ImGui::IsMouseDragging(ImGuiMouseButton_Middle, 0.0f)) {
        return true;
    }

    // Cross-platform fallback that works well on laptops/trackpads
    if (ImGui::IsMouseDragging(ImGuiMouseButton_Left, 0.0f) && IsSpaceDown(window)) {
        return true;
    }

#if defined(__APPLE__)
    // Optional Mac-specific fallback
    if (ImGui::IsMouseDragging(ImGuiMouseButton_Left, 0.0f) && IsCommandDown(window)) {
        return true;
    }
#endif

    return false;
}

}    // namespace

void DetailUI::text(const std::string &s)
{
    ImGui::TextUnformatted(s.c_str());
}

void DetailUI::separator()
{
    ImGui::Separator();
}

// Setup and teardown for ImGUI
struct TreeViewer::Impl {
    Impl(const ViewerConfig viewer_cfg, const TreeLayoutConfig &tree_cfg)
        : viewer_config(std::move(viewer_cfg)), tree_config(std::move(tree_cfg))
    {
        glfwSetErrorCallback(glfw_error_callback);
        if (!glfwInit()) {
            constexpr auto error_msg = "Failed to initialize GLFW";
            spdlog::error(error_msg);
            throw std::runtime_error(error_msg);
        }

        // Decide GL+GLSL versions
#if defined(__APPLE__)
        // Needed on macOS for a modern OpenGL context.
        glsl_version = "#version 150";
        glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
        glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 2);
        glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
        glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
#else
        glsl_version = "#version 330";
        glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
        glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
        glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
#endif

        // Create window with graphics context
        window = glfwCreateWindow(viewer_config.width, viewer_config.height, "Tree Visualizer", nullptr, nullptr);
        if (!window) {
            glfwTerminate();
            constexpr auto error_msg = "Failed to create GLFW window";
            spdlog::error(error_msg);
            throw std::runtime_error(error_msg);
        }

        glfwMakeContextCurrent(window);
        glfwSwapInterval(1);    // Enable vsync

        // Setup Dear ImGui context
        IMGUI_CHECKVERSION();
        ImGui::CreateContext();
        ImGuiIO &io = ImGui::GetIO();
        io.ConfigFlags |= ImGuiConfigFlags_DockingEnable;

        // Setup Dear ImGui style
        if (viewer_config.dark_mode) {
            ImGui::StyleColorsDark();
        } else {
            ImGui::StyleColorsLight();
        }

        // Setup Platform/Renderer backends
        ImGui_ImplGlfw_InitForOpenGL(window, true);
        ImGui_ImplOpenGL3_Init(glsl_version);
    }

    // No copying
    Impl(const Impl &) = delete;
    Impl(Impl &&) = delete;
    auto operator=(const Impl &) -> Impl & = delete;
    auto operator=(Impl &&) -> Impl & = delete;

    ~Impl()
    {
        if (image_texture.id != 0) {
            glDeleteTextures(1, &image_texture.id);
        }
        ImGui_ImplOpenGL3_Shutdown();
        ImGui_ImplGlfw_Shutdown();
        ImGui::DestroyContext();

        if (window) {
            glfwDestroyWindow(window);
        }
        glfwTerminate();
    }

    void begin_frame()
    {
        glfwPollEvents();

        // Start the Dear ImGui frame
        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();
    }

    void end_frame()
    {
        // Rendering
        ImGui::Render();
        int display_w = 0;
        int display_h = 0;
        glfwGetFramebufferSize(window, &display_w, &display_h);

        glViewport(0, 0, display_w, display_h);
        glClearColor(CLEAR_COLOR.r, CLEAR_COLOR.g, CLEAR_COLOR.b, CLEAR_COLOR.a);
        glClear(GL_COLOR_BUFFER_BIT);
        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

        // Swap buffers
        glfwSwapBuffers(window);
    }

    [[nodiscard]] auto is_open() const -> bool
    {
        return window && !glfwWindowShouldClose(window);
    }

    [[nodiscard]] auto get_step_amount() const -> int
    {
        return step_amount;
    }

    [[nodiscard]] auto reset_clicked() const -> bool
    {
        return is_reset_clicked;
    }

    // Build the visual tree from the prepared tree (templated -> generic node type)
    void build_visual_tree(const PreparedTree &prepared)
    {
        // Set nodes
        VisualTree visual_tree = {};
        visual_tree.nodes.reserve(prepared.nodes.size());
        for (const PreparedNode &n : prepared.nodes) {
            visual_tree.nodes.push_back(
                VisualNode{
                .id = n.id,
                .parent_id = n.parent_id,
                .action_taken = n.action_taken,
                .label = n.label,
                .source_index = n.source_index
                }
            );
        }

#ifdef __GLIBCXX__
        for (auto &&[i, node] : std::views::enumerate(visual_tree.nodes)) {
#else
        for (std::size_t i = 0; i < visual_tree.nodes.size(); ++i) {
            auto &node = visual_tree.nodes[i];
#endif
            visual_tree.index_by_id.emplace(node.id, i);
        }

        // Set tree edges
#ifdef __GLIBCXX__
        for (auto &&[child_idx, child] : std::views::enumerate(visual_tree.nodes)) {
#else
        for (std::size_t child_idx = 0; child_idx < visual_tree.nodes.size(); ++child_idx) {
            auto &child = visual_tree.nodes[child_idx];
#endif
            // Root is not a child of any node
            if (!child.parent_id) {
                continue;
            }

            auto par_idx = visual_tree.index_by_id.at(*child.parent_id);
            visual_tree.edges.emplace_back(par_idx, child_idx, child.action_taken);
        }

        // Build the minimal structural input for the layout module
        std::vector<detail::LayoutInputNode> layout_nodes;
        layout_nodes.reserve(visual_tree.nodes.size());

        for (const auto &node : visual_tree.nodes) {
            layout_nodes.push_back(detail::LayoutInputNode{.id = node.id, .parent_id = node.parent_id});
        }

        // Compute or reuse cached layout positions
        detail::compute_tree_layout(layout_nodes, layout_cache, tree_config);

        // Apply cached positions back onto the visual nodes
        for (auto &node : visual_tree.nodes) {
            assert(layout_cache.positions_by_id.contains(node.id));
            const auto &cached_pos = layout_cache.positions_by_id.at(node.id);
            node.x = cached_pos.x;
            node.y = cached_pos.y;
        }
        visual_tree_cache.tree = std::move(visual_tree);
    }

    void build_tree_if_needed(const PreparedTree &prepared)
    {
        const std::uint64_t structure_fp = compute_prepared_tree_fingerprint(prepared);

        // Reuse cache if possible
        if (visual_tree_cache.valid && visual_tree_cache.structure_fingerprint == structure_fp) {
            return;
        }

        // Structure changed, rebuild everything.
        build_visual_tree(prepared);

        visual_tree_cache.structure_fingerprint = structure_fp;
        visual_tree_cache.valid = true;
    }

    void SetupDockspaceLayout()
    {
        ImGuiViewport *viewport = ImGui::GetMainViewport();
        ImGuiID dockspace_id = ImGui::GetID("MainDockspace");

        // Only create the default layout if this dockspace has not been built yet.
        if (ImGui::DockBuilderGetNode(dockspace_id) == nullptr) {
            ImGui::DockBuilderAddNode(dockspace_id, ImGuiDockNodeFlags_DockSpace);
            ImGui::DockBuilderSetNodeSize(dockspace_id, viewport->WorkSize);

            ImGuiID dock_id_tree = dockspace_id;
            ImGuiID dock_id_details = 0;

            ImGui::DockBuilderSplitNode(
                dockspace_id,
                ImGuiDir_Right,
                viewer_config.window_split_percentage,
                &dock_id_details,
                &dock_id_tree
            );

            ImGui::DockBuilderDockWindow("Tree", dock_id_tree);
            ImGui::DockBuilderDockWindow("Node Info", dock_id_details);

            ImGui::DockBuilderFinish(dockspace_id);
        }

        ImGui::DockSpaceOverViewport(dockspace_id, viewport);
    }

    void draw_tree_panel()
    {
        // NOLINTBEGIN(*-magic-numbers)
        ImGui::Begin("Tree");

        ImGui::Text("Left click: select node");    // NOLINT(*-type-vararg)
        ImGui::Text("Middle drag: pan");           // NOLINT(*-type-vararg)
        ImGui::Text("Mouse wheel: zoom");          // NOLINT(*-type-vararg)
        step_amount = 0;
        if (ImGui::Button("Step (1)")) {
            step_amount = 1;
        }
        ImGui::SameLine();
        if (ImGui::Button("Step (5)")) {
            step_amount = 5;
        }
        ImGui::SameLine();
        if (ImGui::Button("Step (10)")) {
            step_amount = 10;
        }
        ImGui::SameLine();
        if (ImGui::Button("Step (100)")) {
            step_amount = 100;
        }
        ImGui::SameLine();
        if (ImGui::Button("Step (1000)")) {
            step_amount = 1000;
        }
        ImGui::SameLine();
        is_reset_clicked = false;
        if (ImGui::Button("Reset Tree")) {
            is_reset_clicked = true;
            visual_tree_cache.clear();
            selected_id.reset();
            clear_cached_image();
            pan = {0.0f, 0.0f};
            zoom = 1.0f;
        }
        ImGui::SameLine();
        if (ImGui::Button("Reset View")) {
            pan = {0.0f, 0.0f};
            zoom = 1.0f;
        }
        ImGui::Separator();

        VisualTree &visual_tree = visual_tree_cache.tree;

        ImVec2 canvas_p0 = ImGui::GetCursorScreenPos();
        ImVec2 canvas_sz = ImGui::GetContentRegionAvail();
        if (canvas_sz.x < 50.0f) canvas_sz.x = 50.0f;
        if (canvas_sz.y < 50.0f) canvas_sz.y = 50.0f;
        ImVec2 canvas_p1(canvas_p0.x + canvas_sz.x, canvas_p0.y + canvas_sz.y);

        ImDrawList *draw_list = ImGui::GetWindowDrawList();
        draw_list->AddRectFilled(canvas_p0, canvas_p1, IM_COL32(45, 45, 50, 255));
        draw_list->AddRect(canvas_p0, canvas_p1, IM_COL32(255, 255, 255, 255));

        ImGui::InvisibleButton(
            "tree_canvas",
            canvas_sz,
            ImGuiButtonFlags_MouseButtonLeft | ImGuiButtonFlags_MouseButtonMiddle
        );

        const bool hovered = ImGui::IsItemHovered();
        const bool active = ImGui::IsItemActive();
        ImGuiIO &io = ImGui::GetIO();

        if (hovered && io.MouseWheel != 0.0f) {
            float next_zoom = zoom * (io.MouseWheel > 0.0f ? 1.1f : 0.9f);
            zoom = (next_zoom < 0.2f) ? 0.2f : (next_zoom > 4.0f ? 4.0f : next_zoom);
        }

        // Middle mouse button drag
        if (active && WantPanDrag(window)) {
            pan.x += io.MouseDelta.x;
            pan.y += io.MouseDelta.y;
        }

        auto to_screen = [&](float x, float y) -> ImVec2 {
            return {canvas_p0.x + pan.x + x * zoom, canvas_p0.y + pan.y + y * zoom};
        };

        // Clip all drawings to the canvas box
        draw_list->PushClipRect(canvas_p0, canvas_p1, true);

        // Draw edges first so they appear behind nodes.
        for (const auto &[parent_i, child_i, action] : visual_tree.edges) {
            const auto &p = visual_tree.nodes[parent_i];
            const auto &c = visual_tree.nodes[child_i];
            ImVec2 p0 = to_screen(p.x, p.y);
            ImVec2 p1 = to_screen(c.x, c.y);
            draw_list->AddLine(p0, p1, IM_COL32(180, 180, 180, 255), 2.0f);
            // draw the actione dge label
            if (zoom >= 0.4) {
                // Midpoint of the edge.
                ImVec2 mid{0.5f * (p0.x + p1.x), 0.5f * (p0.y + p1.y)};

                // Direction of the line.
                float dx = p1.x - p0.x;
                float dy = p1.y - p0.y;
                float len = std::sqrt(dx * dx + dy * dy);

                // Unit perpendicular vector to offset the text off the line a bit.
                ImVec2 normal{0.0f, -1.0f};
                if (len > 1e-5f) {
                    normal.x = -dy / len;
                    normal.y = dx / len;
                }

                // Move the label a little away from the line.
                constexpr float edge_label_offset = 10.0f;
                ImVec2 text_pos{mid.x + normal.x * edge_label_offset, mid.y + normal.y * edge_label_offset};

                // Measure text so we can center it around text_pos.
                std::string edge_label = std::to_string(action);
                ImVec2 text_size = ImGui::CalcTextSize(edge_label.c_str());

                text_pos.x -= 0.5f * text_size.x;
                text_pos.y -= 0.5f * text_size.y;

                // Optional small background behind the text for readability.
                ImVec2 bg_min{text_pos.x - 3.0f, text_pos.y - 2.0f};
                ImVec2 bg_max{text_pos.x + text_size.x + 3.0f, text_pos.y + text_size.y + 2.0f};

                draw_list->AddRectFilled(bg_min, bg_max, IM_COL32(35, 35, 40, 220), 3.0f);

                draw_list->AddText(text_pos, IM_COL32(255, 255, 255, 255), edge_label.c_str());
            }
        }

        // Draw nodes and do hit-testing for clicked node
        for (const auto &node : visual_tree.nodes) {
            ImVec2 p = to_screen(node.x, node.y);
            float r = 20.0f * zoom;

            ImU32 color =
                (selected_id && *selected_id == node.id) ? IM_COL32(255, 200, 0, 255) : IM_COL32(100, 180, 255, 255);

            draw_list->AddCircleFilled(p, r, color, 24);
            if (zoom >= 0.25) {
                draw_list->AddText(ImVec2(p.x + r + 6.0f, p.y - 8.0f), IM_COL32_WHITE, node.label.c_str());
            }

            if (hovered && ImGui::IsMouseClicked(ImGuiMouseButton_Left)) {
                const ImVec2 m = io.MousePos;
                const float dx = m.x - p.x;
                const float dy = m.y - p.y;
                if (dx * dx + dy * dy <= r * r) {
                    selected_id = node.id;
                }
            }
        }
        draw_list->PopClipRect();

        ImGui::End();
        // NOLINTEND(*-magic-numbers)
    }

    void draw_detail_panel(const DetailCallback &detail_callback, const ImageCallback &image_callback)
    {
        ImGui::Begin("Node Info");

        // No node details if no node is clicked
        if (!selected_id) {
            ImGui::Text("Click a node.");    // NOLINT(*-type-vararg)
            ImGui::End();
            return;
        }

        // Selected node no longer exists (shouldn't happen)
        VisualTree &visual_tree = visual_tree_cache.tree;
        if (!visual_tree.index_by_id.contains(*selected_id)) {
            ImGui::Text("Selected node no longer exists.");    // NOLINT(*-type-vararg)
            clear_cached_image();
            ImGui::End();
            return;
        }

        const auto selected_idx = visual_tree.index_by_id.at(*selected_id);
        const VisualNode &node = visual_tree.nodes[selected_idx];

        // Selection changed, so get image and cache
        if (!cached_image_source_index || *cached_image_source_index != node.source_index) {
            cached_image = image_callback(node.source_index);
            cached_image_source_index = node.source_index;
        }

        // Draw the cached image
        const auto [img_height, img_width] = cached_image.shape;
        if (img_width > 0 && img_height > 0 && !cached_image.data.empty()) {
            const std::size_t expected_size =
                static_cast<std::size_t>(img_width) * static_cast<std::size_t>(img_height) * 3;
            ImGui::Separator();
            ImGui::Text("Observation");    // NOLINT(*-type-vararg)
            if (cached_image.data.size() != expected_size) {
                ImGui::Text("Observation data size mismatch.");    // NOLINT(*-type-vararg)
            } else {
                // Rescale to fit
                ImVec2 avail = ImGui::GetContentRegionAvail();
                if (avail.x > 0.0f && avail.y > 0.0f) {
                    const float scale =
                        std::min(avail.x / static_cast<float>(img_width), avail.y / static_cast<float>(img_height));
                    if (scale > 0.0f) {
                        const ImVec2 image_size{
                            static_cast<float>(img_width) * scale,
                            static_cast<float>(img_height) * scale
                        };
                        upload_image_texture(cached_image.data, img_width, img_height);
                        ImGui::Image(
                            static_cast<ImTextureID>(static_cast<std::uintptr_t>(image_texture.id)),
                            image_size
                        );
                    }
                }
            }
        }

        // Draw node details
        DetailUI ui;
        detail_callback(node.source_index, ui);

        ImGui::End();
    }

    void ensure_image_texture()
    {
        if (image_texture.id != 0) {
            return;
        }

        glGenTextures(1, &image_texture.id);
        glBindTexture(GL_TEXTURE_2D, image_texture.id);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    }

    void upload_image_texture(const std::vector<std::uint8_t> &pixels, int width, int height)
    {
        if (width <= 0 || height <= 0) {
            return;
        }

        ensure_image_texture();
        glBindTexture(GL_TEXTURE_2D, image_texture.id);
        glPixelStorei(GL_UNPACK_ALIGNMENT, 1);

        if (image_texture.width != width || image_texture.height != height) {
            glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, width, height, 0, GL_RGB, GL_UNSIGNED_BYTE, pixels.data());
            image_texture.width = width;
            image_texture.height = height;
        } else {
            glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, width, height, GL_RGB, GL_UNSIGNED_BYTE, pixels.data());
        }
    }

    void clear_cached_image()
    {
        cached_image = {};
        cached_image_source_index.reset();
        image_texture.width = 0;
        image_texture.height = 0;
    }

    ViewerConfig viewer_config;
    TreeLayoutConfig tree_config;

    // Platform / graphics state
    GLFWwindow *window = nullptr;
    const char *glsl_version = nullptr;

    // Interaction state
    ImVec2 pan{0.0f, 0.0f};
    float zoom = 1.0f;
    std::optional<int> selected_id;

    // Current tree to draw
    VisualTreeCache visual_tree_cache;
    detail::TreeLayoutCache layout_cache;

    ImageTexture image_texture;
    ImageData cached_image;
    std::optional<std::size_t> cached_image_source_index;

    int step_amount = 0;
    bool is_reset_clicked = false;
};

TreeViewer::TreeViewer(const ViewerConfig &viewer_config, const TreeLayoutConfig &tree_config)
    : impl_(std::make_unique<Impl>(viewer_config, tree_config))
{}
TreeViewer::~TreeViewer() = default;

void TreeViewer::render_prepared(
    const PreparedTree &tree,
    const DetailCallback &detail_callback,
    const ImageCallback &image_cb
)
{
    impl_->build_visual_tree(tree);
    impl_->begin_frame();

    impl_->SetupDockspaceLayout();
    impl_->draw_tree_panel();
    impl_->draw_detail_panel(detail_callback, image_cb);

    impl_->end_frame();
}

auto TreeViewer::is_open() const -> bool
{
    return impl_->is_open();
}

auto TreeViewer::step_amount() const -> int
{
    return impl_->get_step_amount();
}

auto TreeViewer::reset_clicked() const -> bool
{
    return impl_->reset_clicked();
}

}    // namespace libpts::treeviz
