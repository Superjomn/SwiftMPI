#pragma once
#include "../utils/all.h"
using google::dense_hash_map;
/**
 * \brief Base method and data structure of parameter.
 *
 * # Data types
 * * @param Param Server-side parameter type
 * * @param Grad Server-side gradient type, optional,  to support AdaGrad
 * * @param ParamCacheT Worker-side parameter type, just a cache
 * * @param GradCacheT Worker-side gradient type, to support local iterative updating 
 *
 * # Data operation
 * to make the framework run smoothly, some operations between the data types above.
 *
 * ## Param and Grad
 * * update: update Param from Grad
 *
 * ## Param to ParamCacheT
 * ### Param to BinaryBuffer
 * Param be transferred to BinaryBuffer, and send in network to Worker
 * ### BinaryBuffer to ParamCacheT
 * BinaryBuffer be transferred to ParamCacheT
 *
 * ## GradCacheT to Grad (to support AdaGrad, optionally)
 * ### GradCacheT to BinaryBuffer
 * ### BinaryBuffer to Grad
 *
 * ## GradCacheT to Param (no AdaGrad, optionally)
 * ### GradCacheT to BinaryBuffer
 * ### BinaryBuffer to Param
 */
namespace swift_snails {
struct SerialData {
    /**
     * @brief Param to BinaryBuffer
     * @warning BinaryBuffer will be transmitted by network, better to drop unneeded data
     */
    BinaryBuffer& operator<< (BinaryBuffer &bb) {
        serize_out(bb);
        return bb;
    }

protected:
    virtual void serize_out(BinaryBuffer &buffer) = 0;
};
/**
 * @brief AdaGrad suppport
 *
 * @warning copy construct sould be defined
 */
template<class Value>
struct Grad : public SerialData {
    typedef Value val_t;
    /**
     * @brief accumulate grad
     *
     * init with Value support
     */
    virtual void init() = 0;
    virtual void accu(const val_t &grad) = 0;
    /**
     * @brief norm gradient
     *
     * Param use this to update
     */
    virtual val_t grad() const = 0;
};  // end struct Grad
/**
 * @brief Basic definition for parameter
 */
template<class Value, class GradVal>
struct Param : public SerialData {
    typedef Value val_t;
    typedef Grad<GradVal> grad_t;

    virtual void assign(const val_t &val) = 0;
    virtual void init(bool random = false) = 0;
    /**
     * update Param from Grad
     */
    virtual void update(const grad_t &grad) = 0;
    // to support PULL
    virtual val_t& data() = 0;
    virtual const val_t& data() const = 0;
};  // end struct Param
/**
 * @brief local parameter type
 *
 * @param Value actural data type without operation
 */
template<class Value>
struct ParamCacheT : public SerialData {
    typedef Value val_t;

    virtual void init(bool random = false) = 0;
    virtual void assign(const val_t& val) = 0;
};
/**
 * @brief local gradient cache 
 * just save the gradient to update to Server
 * @param Value actural data type without operation
 */
template<class Value>
struct GradCacheT : public SerialData {
    typedef Value val_t;

    virtual void init() = 0;
    virtual void accu (val_t &grad) = 0;
    /** @brief get normalized grad */
    virtual val_t grad() const;
};
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
            _params[key] = param_t();
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
    dense_hash_map<key_t, param_t>& params() {
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
    dense_hash_map<key_t, param_t> _params;
    // gradient cache
    dense_hash_map<key_t, grad_t> _grads;
    std::set<key_t> _local_keys;
};

};  // end namespace swift_snails
