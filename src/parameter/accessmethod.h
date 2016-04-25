#pragma once
#include "../utils/all.h"
namespace swift_snails {
/**
 * @brief Base definition of parameter pull methods
 */
template <typename Key, typename Param, typename PullVal>
class PullAccessMethod {
public:
  typedef Key key_t;
  typedef Param param_t;
  typedef PullVal pull_t;
  /**
   * @brief assign an initial value to param
   */
  virtual void init_param(const key_t &key, param_t &param) = 0;
  /**
   * @brief assign param to val
   */
  virtual void get_pull_value(const key_t &key, const param_t &param,
                              pull_t &val) = 0;
}; // end class PullAccessMethod

template <typename Key, typename Param, typename Grad> class PushAccessMethod {
public:
  typedef Key key_t;
  typedef Param param_t;
  typedef Grad grad_t;
  /**
   * @brief update Server-side parameter with grad
   */
  virtual void apply_push_value(const key_t &key, param_t &param,
                                const grad_t &grad) = 0;

}; // end class PushAccessMethod
   /**
    * @brief Server-side operation agent
    *
    * Pull: worker parameter query request.
    *
    * @param Table subclass of SparseTable
    * @param AccessMethod Server-side operation on parameters
    */
template <typename Table, typename AccessMethod> class PullAccessAgent {
public:
  typedef Table table_t;
  typedef typename Table::key_t key_t;
  typedef typename Table::value_t value_t;

  typedef AccessMethod access_method_t;
  typedef typename AccessMethod::pull_t pull_val_t;
  typedef typename AccessMethod::param_t pull_param_t;

  explicit PullAccessAgent() {}
  void init(table_t &table) { _table = &table; }

  explicit PullAccessAgent(table_t &table) : _table(&table) {}

  int to_shard_id(const key_t &key) { return _table->to_shard_id(key); }
  /**
   * Server-side query parameter
   */
  void get_pull_value(const key_t &key, pull_val_t &val) {
    pull_param_t param;
    if (!_table->find(key, param)) {
      _access_method.init_param(key, param);
      _table->assign(key, param);
    }
    _access_method.get_pull_value(key, param, val);
  }
  /**
   * @brief Worker-side get pull value
   */
  void apply_pull_value(const key_t &key, pull_param_t &param,
                        const pull_val_t &val) {
    _access_method.apply_pull_value(key, param, val);
  }

private:
  table_t *_table;
  AccessMethod _access_method;
}; // class AccessAgent
   /**
    * @brief Server-side push agent
    */
template <typename Table, typename AccessMethod> class PushAccessAgent {
public:
  typedef Table table_t;
  typedef typename Table::key_t key_t;
  typedef typename Table::value_t value_t;

  typedef typename AccessMethod::grad_t push_val_t;
  typedef typename AccessMethod::param_t push_param_t;

  explicit PushAccessAgent() {}
  void init(table_t &table) { _table = &table; }

  explicit PushAccessAgent(table_t &table) : _table(&table) {}
  /**
   * @brief update parameters with the value from remote worker nodes
   */
  void apply_push_value(const key_t &key, const push_val_t &push_val) {
    // RAW_LOG_INFO ("apply_push key:\t%d", key);
    push_param_t *param = nullptr;
    /*
    if (!_table ->find(key, param)) {
        RAW_LOG_ERROR ("skip unknown key:\t%d", key);
        return;
    }
    */
    // TODO improve this in fix mode?
    CHECK(_table->find(key, param)) << "new key should be inited before:\t"
                                    << key;
    CHECK_NOTNULL(param);
    /*
    DLOG(INFO) << "to apply push val: key:\t" << key
               << "\tparam\t" << *param
               << "\tpush_val\t" << push_val;
    */
    _access_method.apply_push_value(key, *param, push_val);
  }

private:
  table_t *_table = nullptr;
  AccessMethod _access_method;

}; // class PushAccessAgent

template <class Key, class Value>
SparseTable<Key, Value> &global_sparse_table() {
  static SparseTable<Key, Value> table;
  return table;
}

template <typename Table, typename AccessMethod>
auto make_pull_access(Table &table)
    -> std::unique_ptr<PullAccessAgent<Table, AccessMethod>> {
  AccessMethod method;
  std::unique_ptr<PullAccessAgent<Table, AccessMethod>> res(
      new PullAccessAgent<Table, AccessMethod>(table));
  return std::move(res);
}

template <typename Table, typename AccessMethod>
auto make_push_access(Table &table)
    -> std::unique_ptr<PushAccessAgent<Table, AccessMethod>> {
  AccessMethod method;
  std::unique_ptr<PushAccessAgent<Table, AccessMethod>> res(
      new PushAccessAgent<Table, AccessMethod>(table));
  return std::move(res);
}

}; // end namespace swift_snails
