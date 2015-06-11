#pragma once
#include "../utils/all.h"
#include "hashfrag.h"
#include "worker.h"
#include "server.h"

namespace swift_snails {

template<class WorkerT, class ServerT, class KeyT>
class Cluster {

public:
    Cluster() : 
        _worker( global_worker()),
        _server( global_server<ServerT>())
    { }

    void initialize() {
        init_route();
        init_hashfrag();
    }
    /**
     * @brief cluster finish working
     *
     * @warning should be called after worker's work is finished
     *
     * will:
     *
     * * tell all Servers to output parameter and exit
     * * tell all Workers to exit
     */
    void finalize() {
        global_mpi().barrier();
        // TODO tell workers to exit
        // TODO tell server to output parameters
        // TODO tell server to exit
        // TODO terminate cluster
    }

protected:
    void init_route() {
        LOG(INFO) << "init global route ...";
        _ports.assign(global_mpi().size() * 2, 0); 
        // distribute port
        _ports[global_mpi().rank() * 2] = _worker.transfer().recv_port();  // worker's port
        _ports[global_mpi().rank() * 2 + 1] = _server.transfer().recv_port();  // server's port
        CHECK(0 == MPI_Allgather(MPI_IN_PLACE, 0, MPI_BYTE, &_ports[0], sizeof(int), MPI_BYTE, MPI_COMM_WORLD));
        // init global route
        for(int rank = 0; rank < global_mpi().size(); rank++) {
            std::string ip = std::string(global_mpi().ip(), global_mpi().IP_WIDTH);
            int worker_port = _ports[rank * 2];
            int server_port = _ports[rank * 2 + 1];
            std::string worker_addr, server_addr;
            format_string(worker_addr, "tcp://%s:%d",  ip.c_str(), worker_port);
            format_string(server_addr, "tcp://%s:%d",  ip.c_str(), server_port);
            VLOG(1) << "worker_addr:\t" << worker_addr;
            VLOG(1) << "server_addr:\t" << server_addr;
            global_route().register_node_(false, std::move(worker_addr));
            global_route().register_node_(true, std::move(server_addr));
        }
        global_mpi().barrier();
    }

    void init_hashfrag() {
        LOG(WARNING) << "... init hashfrag";
        // num of server nodes
        global_hashfrag<KeyT>().set_num_nodes(global_mpi().size());
        global_hashfrag<KeyT>().init();

        global_mpi().barrier();
    }


private:
    WorkerT &_worker;
    ServerT &_server;
    std::vector<int> _ports;
};   // end class Cluster


};  // end namespace swift_snails
