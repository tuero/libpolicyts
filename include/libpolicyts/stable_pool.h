// File: stable_pool.h
// Description: Stable storage pool with optional indexed lookup

#ifndef LIBPTS_UTIL_STABLE_POOL_H_
#define LIBPTS_UTIL_STABLE_POOL_H_

#include <absl/container/flat_hash_set.h>

#include <cstddef>
#include <deque>
#include <functional>
#include <utility>

namespace libpts {

// Wrapper around std::deque for stable pointer storage
template <typename T>
class StablePool {
public:
    StablePool() = default;

    /**
     * Add to the storage pool and receive back a pointer to that object
     */
    template <typename... Args>
    auto emplace(Args &&...args) -> T * {
        items_.emplace_back(std::forward<Args>(args)...);
        return &items_.back();
    }

    /**
     * Clear the items in the storage pool
     * @NOTE: This will invalidate all pointers
     */
    void clear() {
        items_.clear();
    }

    /**
     * Get the size of the storage pool
     */
    [[nodiscard]] auto size() const -> std::size_t {
        return items_.size();
    }

private:
    std::deque<T> items_;
};

// Wrapper around StablePool with indexing
template <typename T, typename HashT = std::hash<T>, typename KeyEqualT = std::equal_to<T>>
class PoolWithIndex {
public:
    PoolWithIndex() = default;

    /**
     * Check if a matching item is stored in the pool
     */
    [[nodiscard]] auto contains(const T &value) const -> bool {
        return index_.find(&value) != index_.end();
    }

    /**
     * Get a pointer matching the given the value if exists, nullptr otherwise
     * @NOTE: Callers need to either check for nullptrs or ensure the value exists
     */
    [[nodiscard]] auto get_ptr(const T &value) const -> const T * {
        const auto itr = index_.find(&value);
        return (itr == index_.end()) ? nullptr : *itr;
    }

    /**
     * If the value exists in the pool then a pointer to it is returned.
     * Otherwise, the value is added to the pool and a constant pointer to it is returned
     */
    auto add_or_get(const T &value) -> const T * {
        if (auto *existing = get_ptr(value)) {
            return existing;
        }
        const auto *ptr = pool_.emplace(value);
        index_.insert(ptr);
        return ptr;
    }

    /**
     * If the value exists in the pool then a pointer to it is returned.
     * Otherwise, the value is added to the pool and a constant pointer to it is returned
     */
    auto add_or_get(T &&value) -> const T * {
        if (auto *existing = get_ptr(value)) {
            return existing;
        }
        const auto *ptr = pool_.emplace(std::move(value));
        index_.insert(ptr);
        return ptr;
    }

    /**
     * If a value matching the constructed object from the args exists in the pool then a pointer to it is returned.
     * Otherwise, the value is added to the pool and a constant pointer to it is returned
     */
    template <typename... Args>
    auto emplace_or_get(Args &&...args) -> const T * {
        T value(std::forward<Args>(args)...);
        return add_or_get(std::move(value));
    }

    /**
     * Clear the items in the storage pool
     * @NOTE: This will invalidate all pointers
     */
    void clear() {
        index_.clear();
        pool_.clear();
    }

    /**
     * Get the size of the storage pool
     */
    [[nodiscard]] auto size() const -> std::size_t {
        return pool_.size();
    }

private:
    class ItemPtrHash {
    public:
        auto operator()(const T *item) const -> std::size_t {
            return hasher_(*item);
        }

    private:
        HashT hasher_;
    };

    class ItemPtrEqual {
    public:
        auto operator()(const T *left, const T *right) const -> bool {
            return equal_to_(*left, *right);
        }

    private:
        KeyEqualT equal_to_;
    };

    StablePool<T> pool_;
    absl::flat_hash_set<const T *, ItemPtrHash, ItemPtrEqual> index_;
};

}    // namespace libpts

#endif    // LIBPTS_UTIL_STABLE_POOL_H_
