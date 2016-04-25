#pragma once
#include <random>

namespace swift_snails {
/*
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
*/

/**
 * @brief simple way to generate random int
 *
 * borrowed from word2vec-C from code.google
 */
struct Random {
  Random(unsigned long long seed) { next_random = seed; }

  unsigned long long operator()() {
    next_random = next_random * (unsigned long long)25214903917 + 11;
    return next_random;
  }

  float gen_float() {
    next_float_random = next_float_random * (unsigned long)4903917 + 11;
    return float(next_float_random) / std::numeric_limits<unsigned long>::max();
  }

private:
  unsigned long long next_random = 0;
  unsigned long next_float_random =
      std::numeric_limits<unsigned long>::max() / 2;
};

inline Random &global_random() {
  static Random r(2008);
  return r;
}

}; // end namespace swift_snails
