//
//  ServerWorkerRoute.h
//  SwiftSnails
//
//  Created by Chunwei on 3/16/15.
//  Copyright (c) 2015 Chunwei. All rights reserved.
//
#ifndef SwiftSnails_transfer_framework_ServerWorkerRoute_h_
#define SwiftSnails_transfer_framework_ServerWorkerRoute_h_
#include "Route.h"
namespace swift_snails {

/**
 * \brief Route with server and worker's route support
 */
class ServerWorkerRoute : public BaseRoute {
public:
  // thread-safe
  int register_node_(bool is_server, std::string &&addr) {
    rwlock_write_guard lock(_read_write_lock);
    int id{-1};
    if (is_server) {
      id = ++_server_num;
      _server_ids.push_back(id);
    } else {
      id = id_max_range - ++_worker_num;
      _worker_ids.push_back(id);
    }
    register_node(id, std::move(addr));
    CHECK_GE(id, 0);
    return id;
  }

  virtual void update() {}

  int server_num() const { return _server_num; }
  int worker_num() const { return _worker_num; }
  const std::vector<int> &server_ids() { return _server_ids; }
  const std::vector<int> &worker_ids() { return _worker_ids; }

private:
  int _server_num = 0;
  int _worker_num = 0;
  std::vector<int> _server_ids;
  std::vector<int> _worker_ids;
  const int id_max_range = std::numeric_limits<int>::max();
};

inline ServerWorkerRoute &global_route() {
  static ServerWorkerRoute route;
  return route;
}

}; // end namespace swift_snails
#endif
