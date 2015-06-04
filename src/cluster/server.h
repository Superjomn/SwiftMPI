#pragma once
#include "../utils/all.h"
#include "../transfer/transfer.h"
#include "../transfer/ServerWorkerRoute.h"

namespace swift_snails {

/**
 * base class of server
 */
class ClusterServer {
public:
    ClusterServer() {
        init_transfer();
    }

    Transfer<ServerWorkerRoute>& transfer() {
        return _transfer;
    }

protected:
    void init_transfer() {
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

private:
    Transfer<ServerWorkerRoute> _transfer;
};  // end class Worker

template<class ServerType>
inline ServerType& global_server() {
    static ServerType server;
    return server;
}


};  // end namespace swift_snails
