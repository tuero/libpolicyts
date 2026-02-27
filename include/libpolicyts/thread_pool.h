// File: thread_pool.h
// Description: Simple thread pool class to dispatch threads continuously on input

#ifndef LIBPTS_THREAD_POOL_H_
#define LIBPTS_THREAD_POOL_H_

#include <atomic>
#include <exception>
#include <functional>
#include <mutex>
#include <optional>
#include <queue>
#include <thread>
#include <vector>

namespace libpts {

// Create a thread pool object.
template <typename InputT, typename OutputT>
class ThreadPool {
public:
    ThreadPool() = delete;

    /**
     * Create a thread pool object.
     * @param num_threads Number of threads the pool should run
     */
    ThreadPool(int num_threads)
        : num_threads_(static_cast<std::size_t>(num_threads)) {
        if (num_threads <= 0) {
            throw std::invalid_argument("Expected at least one thread count");
        }
    }

    /**
     * Run the given function on the thread pool.
     * @param func Function to run in parallel, which should match templated arguments for input and output
     * @param inputs Input items for each job, gets passed to the given function
     * @return Vector of results, in order of given jobs during construction
     */
    [[nodiscard]] auto run(std::function<OutputT(InputT)> func, const std::vector<InputT> &inputs)
        -> std::vector<OutputT> {
        // Populate queue
        int id = -1;
        stop_requested_.store(false, std::memory_order_relaxed);
        first_exception_ = nullptr;
        {
            std::queue<QueueItemInput> empty;
            std::swap(queue_input_, empty);
        }
        {
            std::queue<QueueItemOutput> empty;
            std::swap(queue_output_, empty);
        }
        for (auto const &job : inputs) {
            queue_input_.emplace(job, ++id);
        }

        // Start N threads
        threads_.clear();
        threads_.reserve(num_threads_);
        for (std::size_t i = 0; i < num_threads_; ++i) {
            threads_.emplace_back([this, func]() { this->thread_runner(func); });
        }

        // Wait for all to complete
        for (auto &t : threads_) {
            t.join();
        }
        threads_.clear();

        // One of the threads failed
        if (first_exception_) {
            std::rethrow_exception(first_exception_);
        }

        // Compile results, such that the id is in order to match passed order
        std::vector<std::optional<OutputT>> result_slots(inputs.size());
        while (!queue_output_.empty()) {
            const auto result = queue_output_.front();
            queue_output_.pop();
            if (result.id >= 0 && static_cast<std::size_t>(result.id) < result_slots.size()) {
                result_slots[static_cast<std::size_t>(result.id)] = std::move(result.output);
            }
        }
        std::vector<OutputT> results;
        results.reserve(inputs.size());
        for (auto &result : result_slots) {
            if (result) {
                results.push_back(std::move(*result));
            }
        }

        return results;
    }

    /**
     * Run the given function on the thread pool.
     * @param func Function to run in parallel, which should match templated arguments for input and output
     * @param inputs Input items for each job, gets passed to the given function
     * @param workers Number of threads to use
     * @return Vector of results, in order of given jobs during construction
     */
    [[nodiscard]] auto run(std::function<OutputT(InputT)> func, const std::vector<InputT> &inputs, std::size_t workers)
        -> std::vector<OutputT> {
        std::size_t old_count = num_threads_;
        num_threads_ = workers;
        const auto results = run(func, inputs);
        num_threads_ = old_count;
        return results;
    }

private:
    struct QueueItemInput {    // Wrapper for input type with id
        InputT input;
        int id;
    };

    struct QueueItemOutput {    // Wrapper for output type with id
        OutputT output;
        int id;
    };

    // Runner for each thread, runs given function and pulls next item from input jobs if available
    void thread_runner(std::function<OutputT(InputT)> func) {
        while (true) {
            // Stop early if another thread throws
            if (stop_requested_.load(std::memory_order_relaxed)) {
                return;
            }
            std::optional<QueueItemInput> item;
            {
                std::lock_guard<std::mutex> lock(queue_input_m_);

                // Stop early if another thread throws
                // Check again if another throw threads while we were waiting for lock
                if (stop_requested_.load(std::memory_order_relaxed)) {
                    return;
                }

                // Jobs are done, thread can stop
                if (queue_input_.empty()) {
                    break;
                }

                item = queue_input_.front();
                queue_input_.pop();
            }

            // Stop early if another thread throws
            // Final check again before we actually do work
            if (stop_requested_.load(std::memory_order_relaxed)) {
                return;
            }

            // Run job
            try {
                auto result = func(item->input);

                // Store result
                {
                    std::lock_guard<std::mutex> lock(queue_output_m_);
                    queue_output_.emplace(std::move(result), item->id);
                }
            } catch (...) {
                // Thread threw, set stop signal so other threads can early exit
                stop_requested_.store(true, std::memory_order_relaxed);
                std::lock_guard<std::mutex> lock(exception_m_);
                if (!first_exception_) {
                    first_exception_ = std::current_exception();
                }
                return;
            }
        }
    }

    std::size_t num_threads_;             // How many threads in the pool
    std::vector<std::thread> threads_;    // Threads in the pool
    std::queue<QueueItemInput> queue_input_;
    std::queue<QueueItemOutput> queue_output_;
    std::mutex queue_input_m_;
    std::mutex queue_output_m_;
    std::exception_ptr first_exception_;    // Saved exception thrown by child thread
    std::mutex exception_m_;
    std::atomic_bool stop_requested_{false};
};

}    // namespace libpts

#endif    // PTS_THREAD_POOL_H_
