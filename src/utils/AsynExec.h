//
//  AsynExec.h
//  core
//
//  Created by Chunwei on 2/6/15.
//  Copyright (c) 2015 Chunwei. All rights reserved.
//
#pragma once
#include "common.h"
#include "BasicChannel.h"

namespace swift_snails {

/**
 * \brief asynchronous execution framework
 */
class AsynExec : public VirtualObject {
public:
  typedef std::function<void()> task_t;
  typedef BasicChannel<task_t> channel_t;

  explicit AsynExec() {}

  // set thread num and open the channel
  explicit AsynExec(int thread_num) : _thread_num(thread_num) {
    _channel = open();
  }

  /**
   * \fn std::shared_ptr<channel_t> open()
   *  \warning  every `open` should end with an `close()` or there will be
   * memory leak
   */
  std::shared_ptr<channel_t> open() {

    std::shared_ptr<MultiWorker> worker = std::make_shared<MultiWorker>();

    for (int i = 0; i < thread_num(); i++) {

      worker->threads.emplace_back([worker]() { worker->start(); });
    } // for

    return {&worker->channel, [this, worker](channel_t *) {
              worker->channel.close();
              DLOG(WARNING) << "channel to destroy, terminate all threads!";
              for (auto &t : worker->threads) {
                t.join();
                // t.terminate();
              }
            }};
  }
  /**
   * \fn void set_thread_num(int x)
   *  \warning should be called before `open()`
   */
  void set_thread_num(int x) {
    CHECK(x > 0);
    _thread_num = x;
  }
  int thread_num() const { return _thread_num; }
  std::shared_ptr<channel_t> channel() { return _channel; }

private:
  int _thread_num = 0;
  std::vector<std::thread> _threads;
  std::shared_ptr<channel_t> _channel;

  struct MultiWorker {
    std::vector<std::thread> threads;
    std::atomic<int> alive_workers_num{0};
    channel_t channel;

    void start() {
      task_t func;
      bool valid;

      alive_workers_num++;

      DLOG(INFO) << "thread " << std::this_thread::get_id() << " started";
      while ((valid = channel.pop(func))) {
        // RAW_DLOG(INFO,  "%lu job's valid: %d", std::this_thread::get_id(),
        // valid);
        if (!valid) {
          break;
        }
        func();
      }

      alive_workers_num--;
      // LOG(INFO) << "thread " << std::this_thread::get_id() << " exit";
      // RAW_LOG(INFO, "%d threads still alive", int(alive_workers_num));
    }
    ~MultiWorker() {
      for (auto &t : threads) {
        if (t.joinable())
          t.join();
      }
    }
  }; // end struct MultiWorker
};   // class AsynExec

void async_exec(int thread_num, AsynExec::task_t &part_task,
                std::shared_ptr<AsynExec::channel_t> channel) {
  StateBarrier barrier;
  std::atomic<int> _thread_num{thread_num};

  AsynExec::task_t _task = [&part_task, &_thread_num, &barrier] {
    part_task();
    // this task complete
    // RAW_LOG(INFO, ">  asunc_exec one task finished task_finish_cout");
    if (--_thread_num == 0) {
      // RAW_LOG(INFO, ">  async_exec unblock");
      barrier.set_state_valid();
      barrier.try_unblock();
    }
  };

  for (int i = 0; i < thread_num; i++) {
    auto __task = _task;
    channel->push(std::move(__task));
  }
  barrier.block();
}

}; // end namespace swift_snails
