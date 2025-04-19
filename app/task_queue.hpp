#pragma once

#include <memory>
#include <queue>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <atomic>

template <typename Task>
class task_queue
{
    const uint32_t thread_count;
    std::unique_ptr<std::thread[]> threads;
    std::queue<Task> tasks{};
    std::mutex tasks_mutex;
    std::condition_variable condition;
    std::atomic_bool running{true};

    void worker()
    {
        for (;;)
        {
            Task task;

            {
                std::unique_lock<std::mutex> lock(tasks_mutex);
                condition.wait(lock, [&]
                               { return !tasks.empty() || !running; });
                if (!running)
                {
                    return;
                }

                task = std::move(tasks.front());
                tasks.pop();
            }

            task();
        }
    }

public:
    task_queue(const uint32_t thread_count = std::thread::hardware_concurrency())
        : thread_count(thread_count), threads(std::make_unique<std::thread[]>(thread_count))
    {
        for (uint32_t i = 0; i < thread_count; ++i)
        {
            threads[i] = std::thread(&task_queue::worker, this);
        }
    }

    void push_task(const Task &task)
    {
        {
            const std::lock_guard<std::mutex> lock(tasks_mutex);

            if (!running)
            {
                throw std::runtime_error("Cannot schedule new task after shutdown.");
            }

            tasks.push(task);
        }

        condition.notify_one();
    }
    ~task_queue()
    {
        {
            std::lock_guard<std::mutex> lock(tasks_mutex);
            running = false;
        }

        condition.notify_all();

        for (uint32_t i = 0; i < thread_count; ++i)
        {
            threads[i].join();
        }
    }
    size_t size() const
    {
        return tasks.size();
    }
};
