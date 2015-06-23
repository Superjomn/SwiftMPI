#pragma once
#include "../utils/all.h"
namespace swift_snails {
/**
 * @brief shard of SparseTable
 *
 * a SparseTable contains several shards and the key-values will be 
 * splitted to the shards.
 *
 * SparseTable use shards to improve the efficiency of Read-Write Lock.
 *
 * @param Key key
 * @param Value param can be pair of Param and Grad if AdaGrad is used
 *
 * Value's operation should be defined in AccessMethod
 */
template<typename Key, typename Value> 
struct alignas(64) SparseTableShard : public VirtualObject {
public:
    typedef Key     key_t;
    typedef Value   value_t;
    typedef google::dense_hash_map<key_t, value_t> map_t;

    SparseTableShard() {
        data().set_empty_key(std::numeric_limits<key_t>::max());
    }

    bool find(const key_t& key, value_t* &val) {
        rwlock_read_guard lock(_rwlock);
        auto it = data().find(key);
        if (it == data().end()) return false;
        val = &(it->second);
        return true;
    }
    bool find(const key_t& key, value_t &val) {
        rwlock_read_guard lock(_rwlock);
        auto it = data().find(key);
        if (it == data().end()) return false;
        val = it->second;
        return true;
    }

    void assign(const key_t& key, const value_t &val) {
        rwlock_write_guard lock(_rwlock);
        data()[key] = val; 
    }

    index_t size() {
        rwlock_read_guard lock(_rwlock);
        return data().size();
    }
    void set_shard_id( int x) {
        CHECK_GE(x, 0);
        _shard_id =  x;
    }
    int shard_id() const {
        return _shard_id;
    }
    /**
     * @brief output parameters to ostream
     * @warning should define value's output method first
     */
    friend std::ostream& operator<< (std::ostream& os, SparseTableShard &shard)
    {
        rwlock_read_guard lk(shard._rwlock);
        for(auto& item : shard.data() ) {
            os << item.first << "\t";
            os << item.second << std::endl;
        }
        return os;
    }

protected:
    // not thread safe!
    map_t& data() {
        return _data;
    }

private:
    map_t _data;
    int _shard_id = -1;
    RWLock _rwlock;
    //mutable std::mutex _mutex;
};  // struct SparseTableShard
/**
 * @brief container of sparse parameters
 *
 * a SparseTable has several shards to split the storage and operation of 
 * parameters.
 */
template<typename Key, typename Value>
class SparseTable : public VirtualObject {
public:
    typedef Key     key_t;
    typedef Value   value_t;
    typedef SparseTableShard<key_t, value_t> shard_t;

    SparseTable() {
        _shard_num = global_config().get_config("server", "shard_num").to_int32();
        _shards.reset(new shard_t[shard_num()]);
    }

    shard_t &shard(int shard_id) {
        return _shards[shard_id];
    }

    bool find(const key_t &key, value_t* &val) {
        int shard_id = to_shard_id(key);
        return shard(shard_id).find(key, val);
    }

    bool find(const key_t& key, value_t &val) {
        int shard_id = to_shard_id(key);
        return shard(shard_id).find(key, val);
    }

    void assign (const key_t& key, const value_t &val) {
        int shard_id = to_shard_id(key);
        shard(shard_id).assign(key, val);
    }
    /**
     * output parameters to ostream
     */
    void output() {
        for(int i = 0; i < shard_num(); i++) {
            std::cout << shard(i);
        }
    }
    /**
     * output to a local file
     */
    void output(const std::string& path) {
        std::ofstream file(path.c_str(), std::ios::out);
        for(int i = 0; i < shard_num(); i++) {
            file << shard(i);
        }
    }

    index_t size() const {
        index_t res = 0;
        for(int i = 0; i < shard_num(); i ++) {
            auto& shard = _shards[i];
            res += shard.size();
        }
        return res;
    }
    // TODO assign protected
    int to_shard_id(const key_t& key) {
        return get_hash_code(key)  % shard_num();
    }
    int shard_num() const {
        return _shard_num;
    }

private:
    std::unique_ptr<shard_t[]> _shards; 
    int _shard_num = 1;
};  // class SparseTable

};  // end namespace swift_snails
