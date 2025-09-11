#ifndef SIMPLE_THREAD_POOL_H
#define SIMPLE_THREAD_POOL_H

#include <vector>
#include <thread>
#include <future>
#include <queue>
#include <functional>
#include <condition_variable>
#include <atomic>
#include <type_traits>
#include <utility>

class ThreadPool {
public:
    explicit ThreadPool(std::size_t thread_count = std::thread::hardware_concurrency())
        : stop_(false) {
        if (thread_count == 0) thread_count = 1;
        workers_.reserve(thread_count);
        for (std::size_t i = 0; i < thread_count; ++i) {
            workers_.emplace_back([this]() { this->workerLoop(); });
        }
    }

    ~ThreadPool() {
        {
            std::unique_lock<std::mutex> lk(mtx_);
            stop_ = true;
        }
        cv_.notify_all();//通知所有线程退出等待阻塞状态
        for (auto &t: workers_) {
            if (t.joinable()) t.join();
        }
    }

    template<class F, class... Args>
    auto enqueue(F &&f, Args &&... args)
        -> std::future<std::invoke_result_t<F, Args...> > {
        using Ret = std::invoke_result_t<F, Args...>;//推断任务的返回类型

        auto task = std::make_shared<std::packaged_task<Ret()> >(//用std::packaged_task把函数和参数打包，方便异步执行和结果获取
            std::bind(std::forward<F>(f), std::forward<Args>(args)...)
        );
        std::future<Ret> fut = task->get_future();//获取任务的future，用于后续获取结果
        {
            std::unique_lock<std::mutex> lk(mtx_);
            if (stop_) {
                throw std::runtime_error("ThreadPool stopped");
            }
            tasks_.emplace([task]() { (*task)(); });
        }
        cv_.notify_one();//通知一个等待的线程有新任务
        return fut;//返回future，调用者可用它获取任务执行结果
    }

private:
    void workerLoop() {
        while (true) {
            std::function<void()> job;//job作为一个可调用对象（函数、lambda等）的容器；负责承接并执行线程池分配给当前工作线程的具体任务
            {//限定互斥锁，最小化锁的持有时间，
                std::unique_lock<std::mutex> lk(mtx_);
                cv_.wait(lk, [this]() { return stop_ || !tasks_.empty(); }); //线程在条件变量上等待，直到 线程池停止 或有任务可执行
                if (stop_ && tasks_.empty()) return;
                job = std::move(tasks_.front());
                tasks_.pop();
            }//取出任务后立即释放锁
            job();
        }
    }

    std::vector<std::thread> workers_;
    std::queue<std::function<void()> > tasks_;
    std::mutex mtx_;
    std::condition_variable cv_;//用于在线程间同步任务的添加和处理
    std::atomic<bool> stop_;
};

#endif // SIMPLE_THREAD_POOL_H
