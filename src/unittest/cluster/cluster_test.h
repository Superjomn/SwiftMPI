#include <climits>
#include "../../cluster/cluster.h"
#include "gtest/gtest.h"
using namespace std;
using namespace swift_snails;

TEST(cluster, init) {
    using  server_t = ClusterServer<int, int, int, int, 
    global_config().load_conf("./demo.conf");
    global_config().parse();
    Cluster<ClusterWorker, ClusterServer, int> cluster;
}

TEST(cluster, initialize) {
    global_config().load_conf("./demo.conf");
    global_config().parse();
    Cluster<ClusterWorker, ClusterServer, int> cluster;
    cluster.initialize();
}
