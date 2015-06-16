// utils
#include "utils/ConfigParser_test.h"
#include "cluster/cluster_test.h"
#include "parameter/param_test.h"
#include "utils/random_test.h"


int main(int argc, char **argv) {  

    MPI_Init(&argc, &argv);
    testing::InitGoogleTest(&argc, argv);  
    return RUN_ALL_TESTS();  
} 
