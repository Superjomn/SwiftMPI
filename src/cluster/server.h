#pragma once
#include "../utils/all.h"
#include "../transfer/transfer.h"
#include "../transfer/ServerWorkerRoute.h"
#include "../parameter/sparsetable.h"
#include "../parameter/accessmethod.h"
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
        _sparsetable(global_sparse_table<key_t, param_t>()),
        _pull_access(std::move(make_pull_access<table_t, pull_access_t>(_sparsetable))),
        _push_access(std::move(make_push_access<table_t, push_access_t>(_sparsetable)))
    {
        // check init parameters
        CHECK(_pull_access && _push_access) << "access is not inited";
        init_transfer();
        init_pull_method();
        init_push_method();
    }
    /**
     * @brief load parameter from a file
     * used in prediction period
     */
    void load(const std::string& path) {
        std::ifstream file(path.c_str());
        CHECK (file.is_open()) << "[file] " << path << " can't be opened";
        auto& hashfrag = global_hashfrag<key_t>();
        const auto server_id = _transfer.client_id();
        key_t key;
        param_t param;
        while (!file.eof()) {
            file >> key >> param;
            if (hashfrag.to_node_id(key) == server_id) {
                _sparsetable.assign(key, param);
            }
        }
    }
    /**
     * @brief called when worker finish working
     */
    void finalize(const std::string& path="") {
        RAW_LOG(WARNING, "server output parameters");
        if (path.empty()) 
            _sparsetable.output();
        else
            _sparsetable.output(path);

        RAW_LOG(WARNING, "########################################");
        RAW_LOG(WARNING, "     Server [%d] terminate normally", global_mpi().rank());
        RAW_LOG(WARNING, "########################################");
    }

    Transfer<ServerWorkerRoute>& transfer() {
        return _transfer;
    }
    /**
     * @brief to tell whether local node's Server is valid
     */
    bool is_valid() const {
        return _transfer.client_id() >= 0;
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
    std::string listen_addr = global_config().get("server", "listen_addr").to_string();
    int service_thread_num = global_config().get("server", "listen_thread_num").to_int32();
    int async_thread_num = global_config().get("server", "async_exec_num").to_int32();
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
                pull_t val;
                req->cont >> key;
                req->cont >> val;
                req_items.emplace_back(std::move(key), std::move(val));
            }
            // query parameters
            for( auto& item : req_items) {
                key_t& key = item.first;
                pull_t& val = item.second;
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
                //RAW_LOG_INFO ("bb >> key:\t%d", key);
                _push_access->apply_push_value(key, grad);
            }
            rsp.cont << 1234;
        };

    _transfer.message_class().add(
        WORKER_PUSH_REQUEST, 
        std::move(handler)
    );
}

template <typename ServerT>
inline ServerT& global_server() {
    static ServerT server;
    return server;
}


};  // end namespace swift_snails
