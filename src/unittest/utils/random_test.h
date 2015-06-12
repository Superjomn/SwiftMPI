#include "../../utils/all.h"
#include <climits>
#include "gtest/gtest.h"

TEST(random, random) {
    for(int i = 0; i < 50; i++) {
        LOG(INFO) << "random:\t" << global_random()();
        LOG(INFO) << "random float:\t" << global_random().gen_float();
    }
}

