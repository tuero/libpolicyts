add_executable(bfs bfs.cpp)
target_link_libraries(bfs PRIVATE libpolicyts)
target_compile_features(bfs PUBLIC cxx_std_23)

# Build version using torch model
if(LIBPOLICYTS_BUILD_UI_SOURCES)
    add_executable(bfs_tree_viz bfs_tree_viz.cpp)
    target_link_libraries(bfs_tree_viz PRIVATE libpolicyts)
    target_compile_features(bfs_tree_viz PUBLIC cxx_std_23)
endif()
