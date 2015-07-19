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
    { 
        _to_split_worker_server = global_config().get("cluster", "to_split_worker_server").to_int32() > 0;
        if (! global_config().get("cluster", "server_num").empty()) 
            _server_num = global_mpi().size();
        else 
            _server_num = global_config().get("cluster", "server_num").to_int32();

        _worker_num = global_mpi().size();
        if (_to_split_worker_server) 
            _worker_num = _worker_num - _server_num;
        CHECK_GT (_server_num, 0);
        CHECK_GT (_worker_num, 0);
    }

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
    void finalize(const std::string& path = "") {
        global_mpi().barrier();
        // TODO tell workers to exit
        _worker.finalize();
        global_mpi().barrier();
        if (path.empty()) _server.finalize();
            else _server.finalize(path);
        global_mpi().barrier();
        // TODO tell server to output parameters
        // TODO tell server to exit
        // TODO terminate cluster
    }

protected:
    /**
     * init route and worker / server
     *
     * server's id start from 0 and inc
     * worker's id start from mpi.size() and dec
     */
    void init_route() {
        LOG(INFO) << "init global route ...";
        _ports.assign(global_mpi().size() * 2, 0); 
        //_ports.assign(_server_num + _worker_num, 0); 
        // distribute port
        _ports[global_mpi().rank() * 2] = _worker.transfer().recv_port();  // worker's port
        _ports[global_mpi().rank() * 2 + 1] = _server.transfer().recv_port();  // server's port
        const int local_rank = global_mpi().rank();
        /*
        if (to_start_server(local_rank)) 
            _ports[local_rank] = _server.transfer().recv_port();
        if (to_start_worker(local_rank)) 
            _ports[_server_num + _worker_num - local_rank - 1] = _worker.transfer().recv_port();
        */
        CHECK(0 == MPI_Allgather(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, &_ports[0], 2, MPI_INT, MPI_COMM_WORLD));
        int server_id, worker_id;
        // init global route
        for(int rank = 0; rank < global_mpi().size(); rank++) {
            worker_id = server_id = -1;
            std::string ip = std::string(global_mpi().ip(rank), global_mpi().IP_WIDTH);
            int worker_port = _ports[rank * 2];
            int server_port = _ports[rank * 2 + 1];
            std::string worker_addr, server_addr;
            format_string(worker_addr, "tcp://%s:%d",  ip.c_str(), worker_port);
            format_string(server_addr, "tcp://%s:%d",  ip.c_str(), server_port);
            LOG(INFO) << "worker_addr:\t" << worker_addr;
            LOG(INFO) << "server_addr:\t" << server_addr;
            if (to_start_worker(local_rank))
                worker_id = global_route().register_node_(false, std::move(worker_addr));
            if (to_start_server(local_rank))
                server_id = global_route().register_node_(true, std::move(server_addr));
            if (rank == local_rank) {
                DLOG (WARNING) << "init local client_id:\t" << worker_id << "\t" << server_id;
                if (to_start_server(rank))
                    global_worker().transfer().set_client_id(worker_id);
                if (to_start_worker(rank))
                    global_server<ServerT>().transfer().set_client_id(server_id);
            }
        }
        global_mpi().barrier();
    }

    bool to_start_server(int id) {
        CHECK_GE (id, 0);
        CHECK_LT (id, global_mpi().size());
        return id < _server_num;
    }
    bool to_start_worker(int id) {
        CHECK_GE (id, 0);
        CHECK_LT (id, global_mpi().size());
        return (global_mpi().size() - id) <= _worker_num;
    }

    void init_hashfrag() {
        LOG(WARNING) << "... init hashfrag";
        // num of server nodes
        //global_hashfrag<KeyT>().set_num_nodes(global_mpi().size());
        global_hashfrag<KeyT>().set_num_nodes(_server_num);
        global_hashfrag<KeyT>().init();

        global_mpi().barrier();
    }


private:
    WorkerT &_worker;
    ServerT &_server;
    std::vector<int> _ports;
    int _server_num;
    int _worker_num;
    bool _to_split_worker_server {false};
};   // end class Cluster


};  // end namespace swift_snails
