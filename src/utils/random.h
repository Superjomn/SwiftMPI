#pragma once
#include <random>

namespace swift_snails {

inline std::default_random_engine& local_random_engine() {
    struct engine_wrapper_t {
        std::default_random_engine engine;
        engine_wrapper_t() {
            static std::atomic<unsigned long> x(0);
            std::seed_seq sseq = {x++, x++, x++, (unsigned long)time(NULL)};
            engine.seed(sseq);
        }
    };
    static thread_local engine_wrapper_t r;
    return r.engine;
}


};  // end namespace swift_snails
