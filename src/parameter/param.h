#pragma once
#include "../utils/all.h"
/**
 * Base method and data structure of parameter.
 */

namespace swift_snails {

/**
 * \brief Basic definition for parameter
 */
template<class Value>
struct Param {
    typedef Value val_t;

    val_t val;

    virtual assign(const val_t &val) = 0;
    virtual init() = 0;
};  // end struct Param


template<class Value>
struct Grad {
    typedef Value val_t;

    val_t val;
    /**
     * \brief accumulate grad
     */
    virtual inc(const val_t &grad) = 0;
    virtual init() = 0;
};  // end struct Grad





};  // end namespace swift_snails
