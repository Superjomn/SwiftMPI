#include "../utils/mpi.h"
using namespace swift_snails;

int main(int argc, char** argv) 
{
    MPI_Init(&argc, &argv);
    auto& mpi = global_mpi();
    
    std::string ip = std::string(mpi.ip(), mpi.IP_WIDTH);
    LOG(INFO) << "rank:\t" << mpi.rank() << "\tlocal ip:\t" << ip;
    
    //MPI_Finalize();
    return 0;
}
