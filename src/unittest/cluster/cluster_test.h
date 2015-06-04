#include <climits>
#include "../../cluster/cluster.h"
#include "gtest/gtest.h"
using namespace std;
using namespace swift_snails;

TEST(cluster, init) {
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
