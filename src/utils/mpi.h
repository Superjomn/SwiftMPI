#pragma once
#include "common.h"
#include "localenv.h"

namespace swift_snails {

class GlobalMPI : public VirtualObject {
public:
    const int IP_WIDTH = 64;

    GlobalMPI() {
        // get current process's id
        CHECK(0 == MPI_Comm_rank(MPI_COMM_WORLD, &_rank));
        // get number of processes
        CHECK(0 == MPI_Comm_size(MPI_COMM_WORLD, &_size));
        // allreduce ip table
        _ip_table.assign(IP_WIDTH * _size, 0);
        std::string ip = get_local_ip();
        CHECK((int)ip.length() < IP_WIDTH);
        std::strcpy(&_ip_table[IP_WIDTH * _rank], ip.c_str());
        CHECK(0 == MPI_Allgather(MPI_IN_PLACE, 0, MPI_BYTE, &_ip_table[0], IP_WIDTH, MPI_BYTE, MPI_COMM_WORLD));
    }

    static void initialize (int argc, char** argv) {
        LOG (INFO) << "init MPI ...";
        MPI_Init(&argc, &argv);
    }

    ~GlobalMPI() {
        MPI_Finalize();
    }

    int rank() {
        return _rank;
    }

    int size() {
        return _size;
    }
    
    const char* ip() {
        return &_ip_table[rank() * IP_WIDTH];
    }

    const char* ip(int rank) {
        return &_ip_table[rank * IP_WIDTH];
    }

    void barrier() {
        CHECK(0 == MPI_Barrier(MPI_COMM_WORLD));
    }

private:
    int _rank;
    int _size;
    std::vector<char> _ip_table;

};  // end GlobalMPI

/**
 * \fn inline GlobalMPI& global_mpi()
 * \warning should call MPI_Init first 
 */
inline GlobalMPI& global_mpi() {
    static GlobalMPI mpi;
    return mpi;
}

};  // end namespace
