// utils
#include "utils/common_test.h"

int main(int argc, char **argv) {  

    MPI_Init(&argc, &argv);
    testing::InitGoogleTest(&argc, argv);  
    return RUN_ALL_TESTS();  
} 
