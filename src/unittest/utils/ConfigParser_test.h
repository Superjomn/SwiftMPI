#include <climits>
#include "../../utils/ConfigParser.h"
#include "gtest/gtest.h"
using namespace std;
using namespace swift_snails;

TEST(ConfigParser, init) {
    ConfigParser config;
}

TEST(ConfigParser, parse) {
    ConfigParser config("utils/a.conf");
    config.parse();
    
    cout << config;    
}

TEST(ConfigParser, getvalue) {
    ConfigParser config("utils/a.conf");
    config.parse();
    ASSERT_EQ(
        config.get_config("worker", "num_threads").to_int32(), 12);

    ASSERT_EQ(
        config.get_config("server", "num_threads").to_int32(), 32);

}
