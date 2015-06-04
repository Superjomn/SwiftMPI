//
//  common.h
//  SwiftSnails
//
//  Created by Chunwei on 12/2/14.
//  Copyright (c) 2014 Chunwei. All rights reserved.
//
#ifndef SwiftSnails_utils_common_h
#define SwiftSnails_utils_common_h

#define NDEBUG

#include <iostream>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <map>
#include <set>
#include <unordered_set>
#include <vector>
#include <mutex>
#include <atomic>
#include <condition_variable>
#include <memory>
#include <queue>
#include <thread>
#include <random>
#include <fstream>
#include <sstream>
#include <mpi.h>
#include "VirtualObject.h"
#include "glog/logging.h"
#include "glog/raw_logging.h"
#include "zmq.h"

#include "string.h"

namespace swift_snails {
// common types
typedef unsigned char   byte_t;
typedef short           int16_t;
typedef unsigned short  uint16_t;
typedef int             int32_t;
typedef unsigned int    uint32_t;
typedef long long       int64_t;
typedef unsigned long long   uint64_t;
typedef uint32_t        index_t;

typedef std::function<void()> voidf_t;

// for repeat patterns
#define SS_REPEAT1(X) SS_REPEAT_PATTERN(X)
#define SS_REPEAT2(X, args...) SS_REPEAT_PATTERN(X) SS_REPEAT1(args)
#define SS_REPEAT3(X, args...) SS_REPEAT_PATTERN(X) SS_REPEAT2(args)
#define SS_REPEAT4(X, args...) SS_REPEAT_PATTERN(X) SS_REPEAT3(args)
#define SS_REPEAT5(X, args...) SS_REPEAT_PATTERN(X) SS_REPEAT4(args)
#define SS_REPEAT6(X, args...) SS_REPEAT_PATTERN(X) SS_REPEAT5(args)

inline std::mutex& global_fork_mutex() {
    static std::mutex mutex;
    return mutex;
}

// threadsafe popen pclose
inline FILE* guarded_popen(const char* command, const char* type) {
    std::lock_guard<std::mutex> lock(global_fork_mutex());
    return popen(command, type);
}

inline int guarded_pclose(FILE* stream) {
    std::lock_guard<std::mutex> lock(global_fork_mutex());
    return pclose(stream);
}

template<class FUNC, class... ARGS>
auto ignore_signal_call(FUNC func, ARGS&&... args) 
    -> typename std::result_of<FUNC(ARGS...)>::type {
    for (;;) {
        auto err = func(args...);
        if (err < 0 && errno == EINTR) {
            LOG(INFO) << "Signal is caught. Ignored.";
            continue;
        }
        return err;
    }
}

inline void zmq_bind_random_port(const std::string& ip, void* socket, std::string& addr, int& port) {
    for(;;) {
        addr = "";
        port = 1024 + rand() % (65536 - 1024);
        format_string(addr, "tcp://%s:%d", ip.c_str(), port);
        // ATTENTION: fix the wied memory leak
        // add the LOG valhind detect no memory leak, else ...
        LOG(WARNING) << "try addr: " << addr;
        int res = 0;
        PCHECK((res = zmq_bind(socket, addr.c_str()), 
                res == 0 || errno == EADDRINUSE));  // port is already in use
        if(res == 0) break;
    }
}

inline void zmq_send_push_once(void* zmq_ctx, zmq_msg_t* zmg, const std::string& addr) {
    void* sender = nullptr;
    PCHECK(sender = zmq_socket(zmq_ctx, ZMQ_PUSH));
    PCHECK(0 == ignore_signal_call(zmq_connect, sender, addr.c_str()));
    PCHECK(ignore_signal_call(zmq_msg_send, zmg, sender, 0) >= 0);
    PCHECK(0 == zmq_close(sender));
}

// ensure thread to exit normally
class thread_guard {
    std::thread& _t;
public:
    explicit thread_guard(std::thread& t) :
        _t(t)
    { }
    explicit thread_guard(std::thread&& t) : 
        _t(t)
    { }
    explicit thread_guard(thread_guard&& other) :
        _t(other._t)
    { }
    thread_guard(thread_guard const&) = delete;
    thread_guard& operator=(thread_guard const&) = delete;

    void join() {
        CHECK(_t.joinable());
        _t.join();
    }
    ~thread_guard() {
        if(_t.joinable()) _t.join();
    }
};


};
#endif

