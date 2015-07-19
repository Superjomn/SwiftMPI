#pragma once
#include "../utils/all.h"
#include "../transfer/transfer.h"
#include "../transfer/ServerWorkerRoute.h"

namespace swift_snails {

/**
 * base class of worker
 */
class ClusterWorker {
public:
    ClusterWorker() {
        init_transfer();
    }

    Transfer<ServerWorkerRoute>& transfer() {
        return _transfer;
    }
    /**
     * @brief called after worker finish working 
     */
    void finalize() {
        RAW_LOG(WARNING, "########################################");
        RAW_LOG(WARNING, "     Worker [%d] terminate normally", global_mpi().rank());
        RAW_LOG(WARNING, "########################################");
    }
    /**
     * @brief to tell whether local node's Worker is valid
     */
    bool is_valid() const {
        return _transfer.client_id() >= 0;
    }

protected:
    void init_transfer() {
        LOG(WARNING) << "init worker's transfer ...";
        std::string listen_addr = global_config().get("worker", "listen_addr").to_string();
        int service_thread_num = global_config().get("worker", "listen_thread_num").to_int32();
        int async_thread_num = global_config().get("worker", "async_exec_num").to_int32();
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

inline ClusterWorker& global_worker() {
    static ClusterWorker worker;
    return worker;
}


};  // end namespace swift_snails
