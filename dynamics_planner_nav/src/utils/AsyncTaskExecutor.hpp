// AsyncTaskExecutor.hpp - Thread pool for heavy callback tasks
// Allows slow callbacks (vision, global planning) to run without blocking fast ones

#ifndef DYNAMICS_PLANNER_NAV_ASYNC_TASK_EXECUTOR_HPP
#define DYNAMICS_PLANNER_NAV_ASYNC_TASK_EXECUTOR_HPP

#include <thread>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <functional>
#include <vector>
#include <atomic>
#include <ros/ros.h>

class AsyncTaskExecutor {
public:
    explicit AsyncTaskExecutor(int num_threads = 4) : stop_(false) {
        workers_.reserve(num_threads);
        for (int i = 0; i < num_threads; ++i) {
            workers_.emplace_back([this] { this->workerLoop(); });
        }
        ROS_INFO("AsyncTaskExecutor initialized with %d worker threads", num_threads);
    }

    ~AsyncTaskExecutor() {
        {
            std::unique_lock<std::mutex> lock(queue_mutex_);
            stop_ = true;
        }
        condition_.notify_all();
        for (std::thread& worker : workers_) {
            if (worker.joinable()) {
                worker.join();
            }
        }
    }

    // Submit a task to be executed asynchronously
    template<typename Func>
    void submit(Func&& task) {
        {
            std::unique_lock<std::mutex> lock(queue_mutex_);
            tasks_.emplace(std::forward<Func>(task));
        }
        condition_.notify_one();
    }

    // Check if there are pending tasks
    size_t pendingTasks() const {
        std::unique_lock<std::mutex> lock(queue_mutex_);
        return tasks_.size();
    }

private:
    void workerLoop() {
        while (true) {
            std::function<void()> task;
            {
                std::unique_lock<std::mutex> lock(queue_mutex_);
                condition_.wait(lock, [this] { return stop_ || !tasks_.empty(); });

                if (stop_ && tasks_.empty()) {
                    return;
                }

                if (!tasks_.empty()) {
                    task = std::move(tasks_.front());
                    tasks_.pop();
                }
            }

            if (task) {
                task();
            }
        }
    }

    std::vector<std::thread> workers_;
    std::queue<std::function<void()>> tasks_;
    mutable std::mutex queue_mutex_;
    std::condition_variable condition_;
    std::atomic<bool> stop_;
};

#endif // DYNAMICS_PLANNER_NAV_ASYNC_TASK_EXECUTOR_HPP
