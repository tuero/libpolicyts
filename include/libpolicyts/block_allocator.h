// File: block_allocator.h
// Description: Block allocator with stable pointer addresses

#ifndef LIBPTS_UTIL_BLOCK_ALLOCATOR_H_
#define LIBPTS_UTIL_BLOCK_ALLOCATOR_H_

#include <absl/container/flat_hash_set.h>
#include <spdlog/spdlog.h>

#include <cassert>
#include <optional>
#include <type_traits>
#include <vector>

namespace libpts {

template <typename T, typename HashT = std::hash<T>, typename KeyEqualT = std::equal_to<T>>
class BlockAllocator {
public:
    BlockAllocator() = delete;
    BlockAllocator(int allocate_increment, std::optional<T> default_item = std::nullopt)
        : allocate_increment_(allocate_increment), default_item_(std::move(default_item)) {
        if (allocate_increment_ < 1) {
            SPDLOG_ERROR("allocate_increment must be >= 1");
            std::exit(1);
        }
        clear();
    }

    /**
     * Add to the container and receive a pointer to it
     * @note if the item is already stored, a pointer to the previously stored
     * item is returned
     * @param item The item to store
     * @return A pointer to the stored item
     */
    template <typename U>
    auto add(U &&item) -> const T * {
        assert(idx_ < static_cast<int>(blocks_.back().size()) - 1);
        if (has_item(item)) {
            return get_ptr(item);
        }

        blocks_.back()[static_cast<std::size_t>(++idx_)] = std::forward<U>(item);
        const auto item_ptr = &(blocks_.back()[static_cast<std::size_t>(idx_)]);
        items_set_.insert(item_ptr);

        if (idx_ >= allocate_increment_ - 1) {
            add_block();
        }
        return item_ptr;
    }

    /**
     * Check if the given state is already being held
     * @param item State to check
     * @return True if the given state is already held, false otherwise
     */
    [[nodiscard]] auto has_item(const T &item) const -> bool {
        return items_set_.find(&item) != items_set_.end();
    }

    /**
     * Get the pointer to the held state which matches the state being queried
     * @param item State to check
     * @return Pointer to held state if exists, nullptr otherwise
     */
    [[nodiscard]] auto get_ptr(const T &item) const -> const T * {
        const auto itr = items_set_.find(&item);
        return (itr == items_set_.end()) ? nullptr : *itr;
    }

    /**
     * Clear all held items and reset the state
     */
    void clear() {
        idx_ = -1;
        items_set_.clear();
        for (auto &block : blocks_) {
            block.clear();
        }
        blocks_.clear();
        add_block();
    }

private:
    void add_block() {
        if constexpr (std::is_default_constructible_v<T>) {
            if (default_item_) {
                blocks_.push_back(std::vector<T>(allocate_increment_, default_item_.value()));
            } else {
                blocks_.push_back(std::vector<T>(allocate_increment_));
            }
        } else {
            blocks_.push_back(std::vector<T>(static_cast<std::size_t>(allocate_increment_), default_item_.value()));
        }
        idx_ = -1;
    }

    // Required because T may have hash defined, but pointers are stored
    class ItemPtrHash {
    public:
        auto operator()(const T *item) const -> std::size_t {
            return hasher(*item);
        }

    private:
        HashT hasher;
    };

    // Required because T may have equal_to defined, but pointers are stored
    class ItemPtrEqual {
    public:
        auto operator()(const T *left, const T *right) const -> bool {
            return equal_to(*left, *right);
        }

    private:
        KeyEqualT equal_to;
    };

    using ItemSet = absl::flat_hash_set<const T *, ItemPtrHash, ItemPtrEqual>;

    int allocate_increment_;
    int idx_ = -1;
    std::optional<T> default_item_;
    std::vector<std::vector<T>> blocks_;
    ItemSet items_set_;
};

}    // namespace libpts

#endif    // LIBPTS_UTIL_BLOCK_ALLOCATOR_H_
