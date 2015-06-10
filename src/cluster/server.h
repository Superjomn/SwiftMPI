#pragma once
#include "../utils/all.h"
#include "../transfer/transfer.h"
#include "../transfer/ServerWorkerRoute.h"
#include "../parameter/sparsetable.h"
#include "message_classes.h"

namespace swift_snails {

/**
 * @brief base class of server
 * @param Key key
 * @param Param type of Server-side parameter
 * @PullVal type of Worker-side parameter. Pull : Param -> PullVal
 * @Grad type of gradient. Push: Grad -> Param
 * @PullAccessMethod pull method
 * @PushAccessMethod push method
 */
template<typename Key, typename Param, typename PullVal, typename Grad, typename PullAccessMethod, typename PushAccessMethod>
class ClusterServer {
public:
    typedef Key key_t;
    typedef Param param_t;
    typedef PullVal pull_t;
    typedef Grad grad_t;
    typedef SparseTable<key_t, param_t> table_t;
    typedef Transfer<ServerWorkerRoute> transfer_t;
    typedef PushAccessMethod push_access_t;
    typedef PullAccessMethod pull_access_t;

    ClusterServer():
        _sparsetable(global_sparse_table<key_t, param_t>())
    {
        init_transfer();
        init_pull_method();
        init_push_method();
    }
    /**
     * @brief called when worker finish working
     */
    void finalize() {
        RAW_LOG(WARNING, "server output parameters");
        _sparsetable.output();
        RAW_LOG(WARNING, "###################################");
        RAW_LOG(WARNING, "     Server terminate normally");
        RAW_LOG(WARNING, "###################################");
    }

    Transfer<ServerWorkerRoute>& transfer() {
        return _transfer;
    }

protected:
    void init_transfer();
    /**
     * @brief register pull method to message class
     */
    void init_pull_method();
    /**
     * @brief register push method to message class
     */
    void init_push_method();
private:
    Transfer<ServerWorkerRoute> _transfer;
    table_t &_sparsetable;
    std::unique_ptr<PullAccessAgent<table_t, pull_access_t>> _pull_access;
    std::unique_ptr<PushAccessAgent<table_t, push_access_t>> _push_access;
};


template<class ServerType> inline ServerType& global_server();


template<typename Key, typename Param, typename PullVal, typename Grad, typename PullAccessMethod, typename PushAccessMethod>
void \
ClusterServer<Key, Param, PullVal, Grad, PullAccessMethod, PushAccessMethod>::\
init_transfer() {
    LOG(WARNING) << "init server's transfer ...";
    std::string listen_addr = global_config().get_config("server", "listen_addr").to_string();
    int service_thread_num = global_config().get_config("server", "listen_thread_num").to_int32();
    int async_thread_num = global_config().get_config("server", "async_exec_num").to_int32();
    if(!listen_addr.empty()) {
        _transfer.listen(listen_addr);
    } else {
        _transfer.listen();
    }
    _transfer.init_async_channel(async_thread_num);
    _transfer.set_thread_num(service_thread_num);
    _transfer.service_start();
}

template<typename Key, typename Param, typename PullVal, typename Grad, typename PullAccessMethod, typename PushAccessMethod>
void \
ClusterServer<Key, Param, PullVal, Grad, PullAccessMethod, PushAccessMethod>::\
init_pull_method() {
    LOG(INFO) << "server register pull message_class ...";
    transfer_t::msgcls_handler_t handler = \
        [this] (std::shared_ptr<Request> req, Request& rsp) 
        {
            // read request
            std::vector<std::pair<key_t,pull_t>> req_items;
            while(! req->cont.read_finished()) {
                key_t key;
                param_t val;
                req->cont >> key;
                req->cont >> val;
                req_items.emplace_back(std::move(key), std::move(val));
            }
            // query parameters
            for( auto& item : req_items) {
                key_t& key = item.first;
                param_t& val = item.second;
                _pull_access->get_pull_value(key, val);
                // put response
                rsp.cont << key;
                rsp.cont << val;
            }
        };

    _transfer.message_class().add(
        WORKER_PULL_REQUEST, 
        std::move(handler));
}

template<typename Key, typename Param, typename PullVal, typename Grad, typename PullAccessMethod, typename PushAccessMethod>
void \
ClusterServer<Key, Param, PullVal, Grad, PullAccessMethod, PushAccessMethod>::\
init_push_method() {
    LOG(INFO) << "server register push message_class ...";
    transfer_t::msgcls_handler_t handler =  \
        [this] (std::shared_ptr<Request> req,  Request& rsp)
        {
            //std::vector<push_val_t> req_items;
            while(! req->cont.read_finished()) {
                key_t key;
                grad_t grad;
                req->cont >> key;
                req->cont >> grad;
                _push_access->apply_push_value(key, grad);
            }
            rsp.cont << 1234;
        };

    _transfer.message_class().add(
        WORKER_PUSH_REQUEST, 
        std::move(handler)
    );
}

inline ClusterServer& global_server() {
    static ClusterServer server;
    return server;
}


};  // end namespace swift_snails
