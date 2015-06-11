#include <climits>
#include <gtest/gtest.h>
#include "../../utils/all.h"
#include "../../cluster/cluster.h"
#include "../../cluster/server.h"
#include "../../parameter/sparsetable.h"
using namespace swift_snails;

class ClusterTestPullMethod : public BasePullAccessMethod<int, float, float> {
public:
    virtual void init_param(const key_t &key, param_t &param)  {
        param = 0.0;
    }

    virtual void get_pull_value(const key_t &key, const param_t &param, pull_t& val) {
        val = param;
    }
};


class ClusterTestPushMethod : public BasePushAccessMethod<int, float, float> {
public:
    virtual void apply_push_value(const key_t &key, param_t &param, const grad_t &grad) {
        param += grad;
    }
};

/*
TEST(cluster, init) {
    using  server_t = ClusterServer<int, float, float, float, ClusterTestPullMethod, ClusterTestPushMethod>;

    global_config().load_conf("./demo.conf");
    global_config().parse();
    Cluster<ClusterWorker, server_t, int> cluster;
}
*/

TEST(cluster, initialize) {
    using  server_t = ClusterServer<int, float, float, float, ClusterTestPullMethod, ClusterTestPushMethod>;
    global_config().load_conf("./demo.conf");
    global_config().parse();
    Cluster<ClusterWorker, server_t, int> cluster;
    cluster.initialize();
}
