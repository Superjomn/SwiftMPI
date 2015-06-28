#include <iostream>
#include "../../utils/all.h"
#include <climits>
#include "gtest/gtest.h"
using namespace swift_snails;

TEST(string, hash) {
    typedef unsigned long key_t;
    char name[] = "superjom";
    auto code = BKDRHash<key_t>(name);
    LOG(INFO) << "size of key\t" << sizeof(key_t);
    LOG(INFO) << name << "\t" << code;
    LOG(INFO) << "a" << "\t" << BKDRHash<key_t>("a");
    LOG(INFO) << "b" << "\t" << BKDRHash<key_t>("b");
    LOG(INFO) << "而且" << "\t" << BKDRHash<key_t>("而且");
    LOG(INFO) << "2014年2月11日 ... 而且经过这么多年发展" << "\t" << BKDRHash<key_t>("2014年2月11日 ... 而且经过这么多年发展");
}
