#pragma once
#include "../utils/all.h"
/**
 * Base method and data structure of parameter.
 */

namespace swift_snails {

/**
 * @brief Basic definition for parameter
 */
template<class Value>
struct Param {
    typedef Value val_t;

    val_t val;

    virtual assign(const val_t &val) = 0;
    virtual init() = 0;
    virtual init(bool random = false) = 0;
};  // end struct Param


template<class Value>
struct Grad {
    typedef Value val_t;

    val_t val;
    /**
     * @brief accumulate grad
     */
    virtual inc(const val_t &grad) = 0;
    virtual init() = 0;
};  // end struct Grad
/**
 * @brief local parameter cache
 *
 * @param Key key
 * @param Param type of local parameter, subclass of Param
 * @param Grad type of gradient, subclass of Grad
 */
template<typename Key, typename Param, typename Grad>
class LocalParamCache {
public:
    typedef Key key_t;
    typedef Param param_t;
    typedef Grad grad_t;

    explicit LocalParamCache() {
        _params.set_empty_key(std::numeric_limits<key_t>::max());
        _grads.set_empty_key(std::numeric_limits<key_t>::max());
    }

    void init_keys(std::set<key_t> &keys) {
        rwlock_write_guard lk(_rwlock);

        for(auto& key : keys) {
            _params[key] = val_t();
            _grads[key] = grad_t();
        }
    }

    size_t size() const {
        rwlock_read_guard lk(_rwlock);
        return _params.size();
    }
    /**
     * @warning not thread-safe
     */
    dense_hash_map<key_t, val_t>& params() {
        return _params;
    }
    /**
     * @warning not thread-safe
     */
    dense_hash_map<key_t, grad_t>& grads() {
        return _grads;
    }
    RWLock& rwlock() {
        return _rwlock;
    }
    friend std::ostream& operator<< (std::ostream& os, LocalParamCache & cache)
    {
        for(auto& item : cache._params ) {
            os << item.first << "\t";
            os << item.second << std::endl;
        }
        return os;
    }
    std::set<key_t>& local_keys() {
        return _local_keys;
    }

private:
    RWLock _rwlock;
    // parameter cache
    dense_hash_map<key_t, val_t> _params;
    // gradient cache
    dense_hash_map<key_t, grad_t> _grads;
    std::set<key_t> _local_keys;
};



};  // end namespace swift_snails
