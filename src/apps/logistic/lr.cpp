#include "../../swiftmpi.h"
using namespace std;
using namespace swift_snails;

typedef unsigned int lr_key_t;

struct LRParam {
  float val = 0;
  float grad2sum = 0;
};

typedef float LRLocalParam;

struct LRLocalGrad {
  float val = 0;
  int count = 0;

  void reset() {
    val = 0;
    count = 0;
  }
};

std::ostream &operator<<(std::ostream &os, const LRParam &param) {
  os << param.val;
  return os;
}
std::istream &operator>>(std::istream &is, LRParam &param) {
  is >> param.val;
  return is;
}
BinaryBuffer &operator<<(BinaryBuffer &bb, LRLocalGrad &grad) {
  // CHECK_GT(grad.count, 0);
  if (grad.count == 0)
    return bb;
  bb << float(grad.val / grad.count);
  return bb;
}
BinaryBuffer &operator>>(BinaryBuffer &bb, LRLocalGrad &grad) {
  bb >> grad.val;
  grad.count = 1;
  return bb;
}

class LRPullAccessMethod
    : public PullAccessMethod<lr_key_t, LRParam, LRLocalParam> {
public:
  virtual void init_param(const lr_key_t &key, param_t &param) {
    param.val = global_random().gen_float();
  }
  virtual void get_pull_value(const lr_key_t &key, const param_t &param,
                              pull_t &val) {
    val = param.val;
    // RAW_LOG_INFO ( "pull\t%d:%f", key, val);
  }
};

class LRPushAccessMethod
    : public PushAccessMethod<lr_key_t, LRParam, LRLocalGrad> {
public:
  LRPushAccessMethod()
      : initial_learning_rate(
            global_config().get("server", "initial_learning_rate").to_float()) {
  }
  /**
   * grad should be normalized before pushed
   */
  virtual void apply_push_value(const lr_key_t &key, param_t &param,
                                const grad_t &push_val) {
    // RAW_LOG_INFO ("push key:param/grad\t%d:%f/%f/%d", key, param.val,
    // push_val.val, push_val.count);
    param.grad2sum += push_val.val * push_val.val;
    param.val += initial_learning_rate * push_val.val /
                 float(std::sqrt(param.grad2sum + fudge_factor));
  }

private:
  float initial_learning_rate;
  static const float fudge_factor;
};
const float LRPushAccessMethod::fudge_factor = 1e-6;
// LocalParamCache<lr_key_t, LRLocalParam, LRLocalGrad> param_cache;

struct Instance {
  float target;
  std::vector<std::pair<unsigned int, float>> feas;

  void clear() {
    // clear data but not free memory
    feas.clear();
  }
};

std::ostream &operator<<(std::ostream &os, const Instance &ins) {
  os << "instance:\t" << ins.target << "\t";
  for (const auto &item : ins.feas) {
    os << item.first << ":" << item.second << " ";
  }
  os << std::endl;
  return os;
}

bool parse_instance2(const std::string &line, Instance &ins) {
  // RAW_LOG_INFO ("parsing \"%s\"", line.c_str());
  const char *pline = line.c_str();
  ins.feas.clear();
  while ((*pline == ' ') || (*pline == 0))
    pline++;
  if ((*pline == 0) || (*pline == '#'))
    return false;
  float value;
  int nchar, feature;
  if (std::sscanf(pline, "%f%n", &value, &nchar) >= 1) {
    pline += nchar;
    ins.target = value;

    while (std::sscanf(pline, "%d:%f%n", &feature, &value, &nchar) >= 2) {
      pline += nchar;
      ins.feas.emplace_back(feature, value);
    }
    // while ((*pline != 0) && ((*pline == ' ') || (*pline == 9))) pline ++;
    /*if ((*pline != 0) && (*pline != '#'))
        throw "cannot parse line \"" + line + "\" at character " + pline[0];
    */
  } else {
    LOG(ERROR) << "parse line error";
    throw "cannot parse line \"" + line + "\" at character " + pline[0];
  }
  // LOG(INFO) << ins;
  return true;
}

typedef ClusterServer<lr_key_t, LRParam, LRLocalParam, LRLocalGrad,
                      LRPullAccessMethod, LRPushAccessMethod> server_t;

class LR {
public:
  typedef GlobalPullAccess<lr_key_t, LRLocalParam, LRLocalGrad> pull_access_t;
  typedef GlobalPushAccess<lr_key_t, LRLocalParam, LRLocalGrad> push_access_t;
  typedef LocalParamCache<lr_key_t, LRLocalParam, LRLocalGrad> param_cache_t;

  LR(const string &path, int niters)
      : _minibatch(global_config().get("worker", "minibatch").to_int32()),
        _nthreads(global_config().get("worker", "nthreads").to_int32()),
        _pull_access(global_pull_access<lr_key_t, LRLocalParam, LRLocalGrad>()),
        _push_access(global_push_access<lr_key_t, LRLocalParam, LRLocalGrad>()),
        _niters(niters) {
    _path = path;
    CHECK_GT(_path.size(), 0);
    CHECK_GT(_minibatch, 0);
    CHECK_GT(_nthreads, 0);
    CHECK_GT(_niters, 0);
    AsynExec exec(_nthreads);
    _async_channel = exec.open();
  }

  void train() {
    // init server-side parameter
    FILE *file = fopen(_path.c_str(), "rb");
    // init keys
    gather_keys(file);
    RAW_LOG_WARNING("... to init local parameter cache");
    _param_cache.init_keys(_local_keys);

    LOG(WARNING) << "... first pull to init local_param_cache";
    pull();
    LOG(WARNING) << ">>> end pull()";
    global_mpi().barrier();

    std::atomic<int> line_count{0};
    LineFileReader line_reader;
    std::mutex file_mut;
    SpinLock spinlock;
    double total_error{0};
    int nrecords{0};

    // first to init local keys
    // gather_keys(file);

    AsynExec::task_t handler = [this, &line_count, &line_reader, &file,
                                &file_mut, &spinlock, &total_error,
                                &nrecords]() {
      std::string line;
      float error;
      char *cline;
      Instance ins;
      bool parse_res;
      while (true) {
        if (feof(file))
          break;
        {
          std::lock_guard<std::mutex> lk(file_mut);
          cline = line_reader.getline(file);
          if (!cline)
            continue;
          line = std::move(string(cline));
        }
        parse_res = parse_instance2(line, ins);
        if (!parse_res)
          continue;
        // if(ins.feas.size() < 4) continue;
        error = learn_instance(ins);
        total_error += error;
        nrecords++;
        line_count++;
        if (line_count > _minibatch)
          break;
        if (feof(file))
          break;
      }
    };
    LOG(WARNING) << "... to train";
    for (int i = 0; i < _niters; i++) {
      LOG(WARNING) << i << "th train";
      while (true) {
        line_count = 0;
        // rebuild local parameter cache
        // LOG (INFO) << "... gather keys";
        gather_keys(file, _minibatch);
        _param_cache.clear();
        _param_cache.init_keys(_local_keys);
        // LOG (INFO) << "... to pull minibatch";
        pull();
        // LOG (INFO) << "... to multi-thread train";
        async_exec(_nthreads, handler, _async_channel);
        // LOG (INFO) << "... to push minibatch";
        push();
        if (feof(file))
          break;
      }
      LOG(INFO) << nrecords << " records\terror:\t" << total_error / nrecords;
      total_error = 0;
      nrecords = 0;
      // jump to file's beginning
      rewind(file);
    }
    LOG(WARNING) << "finish training ...";
  }

  void predict(const std::string &path, const std::string &out) {
    LOG(WARNING) << "to predict " << path;
    std::atomic<int> line_count{0};
    LineFileReader line_reader;
    std::mutex file_mut;
    SpinLock spinlock;
    double total_error{0};
    int nrecords{0};
    FILE *file = fopen(_path.c_str(), "rb");
    std::ofstream outfile(out.c_str());
    AsynExec::task_t handler = [this, &line_count, &line_reader, &file,
                                &file_mut, &spinlock, &total_error, &nrecords,
                                &outfile]() {
      std::string line;
      float error, predict;
      char *cline;
      Instance ins;
      bool parse_res;
      while (true) {
        if (feof(file))
          break;
        {
          std::lock_guard<std::mutex> lk(file_mut);
          cline = line_reader.getline(file);
          if (!cline)
            continue;
          line = std::move(string(cline));
        }
        parse_res = parse_instance2(line, ins);
        if (!parse_res)
          continue;
        // if(ins.feas.size() < 4) continue;
        error = predict_instance(ins, predict);
        outfile << predict << std::endl;
        total_error += error;
        nrecords++;
        line_count++;
        if (line_count > _minibatch)
          break;
        if (feof(file))
          break;
      }
    };

    while (true) {
      line_count = 0;
      // rebuild local parameter cache
      gather_keys(file, _minibatch);
      _param_cache.clear();
      _param_cache.init_keys(_local_keys);
      pull();
      async_exec(1, handler, _async_channel);
      if (feof(file))
        break;
    }
  }

  void load_param(const std::string &path) {
    global_server<server_t>().load(path);
    global_mpi().barrier();
  }

protected:
  /**
   * @brief gather keys within a minibatch
   * @param file file with fopen
   * @param minibatch size of Mini-batch, no limit if minibatch < 0
   */
  void gather_keys(FILE *file, int minibatch = -1) {
    long cur_pos = ftell(file);
    std::atomic<int> line_count{0};
    LineFileReader line_reader;
    std::mutex file_mut;
    SpinLock spinlock;
    _local_keys.clear();
    // CounterBarrier cbarrier(_nthreads);

    AsynExec::task_t handler = [this, &line_count, &line_reader, &file_mut,
                                &spinlock, minibatch, &file] {
      char *cline = nullptr;
      std::string line;
      Instance ins;
      bool parse_res;
      while (true) {
        if (feof(file))
          break;
        ins.clear();
        {
          std::lock_guard<std::mutex> lk(file_mut);
          cline = line_reader.getline(file);
          if (!cline)
            continue;
          line = std::move(string(cline));
        }
        parse_res = parse_instance2(line, ins);
        if (!parse_res)
          continue;
        // if(ins.feas.size() < 4) continue;
        {
          std::lock_guard<SpinLock> lk(SpinLock);
          for (const auto &item : ins.feas) {
            _local_keys.insert(item.first);
          }
        }
        line_count++;
        if (minibatch > 0 && line_count > minibatch)
          break;
        if (feof(file))
          break;
      }
    };
    async_exec(_nthreads, handler, _async_channel);
    // RAW_LOG(INFO, "collect %d keys", _local_keys.size());
    fseek(file, cur_pos, SEEK_SET);
  }
  /**
   * SGD update
   */
  float learn_instance(const Instance &ins) {
    float sum = 0;
    for (const auto &item : ins.feas) {
      auto param = _param_cache.params()[item.first];
      sum += param * item.second;
    }
    float predict = 1. / (1. + exp(-sum));
    float error = ins.target - predict;
    // update grad
    float grad = 0;
    for (const auto &item : ins.feas) {
      grad = error * item.second;
      _param_cache.grads()[item.first].val += grad;
      // RAW_LOG_INFO( "grad:\t%d:%f", item.first, grad);
      _param_cache.grads()[item.first].count++;
    }
    return error * error;
  }
  float predict_instance(const Instance &ins, float &predict) {
    float sum = 0;
    for (const auto &item : ins.feas) {
      auto param = _param_cache.params()[item.first];
      sum += param * item.second;
    }
    predict = 1. / (1. + exp(-sum));
    float error = ins.target - predict;
    return error;
  }

protected:
  /**
   * query parameters contained in local cache from remote server
   */
  void pull() { _pull_access.pull_with_barrier(_local_keys, _param_cache); }
  /**
   * update server-side parameters with local grad
   */
  void push() {
    _push_access.push_with_barrier(_local_keys, _param_cache);
    _local_keys.clear();
  }

private:
  // dataset path
  string _path;
  int _minibatch;
  int _nthreads;
  int _niters;
  pull_access_t &_pull_access;
  push_access_t &_push_access;
  param_cache_t _param_cache;
  std::unordered_set<lr_key_t> _local_keys;
  std::shared_ptr<AsynExec::channel_t> _async_channel;
};

int main(int argc, char **argv) {
  GlobalMPI::initialize(argc, argv);
  fms::CMDLine cmdline(argc, argv);
  auto C = [&cmdline](const string &key) { return cmdline.getValue(key); };
  // cmdline args
  string param_help = cmdline.registerParameter("help", "this screen");
  string param_mode = cmdline.registerParameter("mode", "train/predict");
  string param_dataset =
      cmdline.registerParameter("dataset", "path of the dataset");
  string param_config =
      cmdline.registerParameter("config", "path of the config file");
  string param_niters = cmdline.registerParameter(
      "niters", "number of iterations (used only in train mode)");
  string param_param_path = cmdline.registerParameter(
      "param_path", "path of parameter (in predict mode)");
  string param_out_prefix =
      cmdline.registerParameter("out_prefix", "path to output predictions");
  // usage infomation
  std::string usage = "\n";
  usage +=
      "-----------------------------------------------------------------\n";
  usage += "An Implementation of Distributed Logistic Regression Algorithm "
           "\nbased on SwiftMPI\n";
  usage +=
      "-----------------------------------------------------------------\n";
  usage += "Author: Chunwei Yan <yanchunwei@outlook.com>\n";
  usage += "\nUsage:\n\n";
  usage += "Train Mode:\n";
  usage += "    <MPI_ARGS>" + std::string(argv[0]) +
           " -mode train -config <path> -niters <number> -dataset <path> "
           "-out_prefix <path prefix>\n";
  usage += "\nPredict Mode:\n";
  usage += "    <MPI_ARGS>" + std::string(argv[0]) +
           " -mode predict -config <path> -dataset <path> -param_path <path> "
           "-out_prefix <string>";
  usage += "\n\n";
  // args check
  auto miss_param_handler = [&cmdline, &usage] {
    LOG(ERROR) << "missing parameter";
    cout << usage << endl;
    cmdline.print_help();
  };
  if (!cmdline.hasParameter(param_mode)) {
    miss_param_handler();
    return 0;
  }
  bool common_args_check = (!cmdline.hasParameter(param_config) ||
                            !cmdline.hasParameter(param_dataset));

  if (C(param_mode) == "train") {
    if (common_args_check || !cmdline.hasParameter(param_niters)) {
      miss_param_handler();
      return 0;
    }
  } else if (C(param_mode) == "predict") {
    if (common_args_check || !cmdline.hasParameter(param_param_path) ||
        !cmdline.hasParameter(param_out_prefix)) {
      miss_param_handler();
      return 0;
    }
  } else {
    miss_param_handler();
    return 0;
  }
  if (cmdline.hasParameter(param_help) || argc == 1) {
    std::cout << usage << std::endl;
    cmdline.print_help();
    return 0;
  }
  // init cluster
  Cluster<ClusterWorker, server_t, lr_key_t> cluster;
  cluster.initialize();
  LR lr(C(param_dataset), stoi(C(param_niters)));
  // to train
  if (C(param_mode) == "train") {
    lr.train();
    std::string out_param_path =
        global_config().get("server", "out_param_prefix").to_string();
    // swift_snails::format_string(out_param_path, "-%d.txt",
    // global_mpi().rank());
    swift_snails::format_string(out_param_path, ".txt");
    RAW_LOG_WARNING("server output parameter to %s", out_param_path.c_str());
    cluster.finalize(out_param_path);

    // to predict
  } else {
    lr.load_param(C(param_param_path));
    string out_prefix = C(param_out_prefix);
    swift_snails::format_string(out_prefix, "-%d.txt", global_mpi().rank());
    lr.predict(C(param_dataset), C(param_out_prefix));
    cluster.finalize();
  }

  LOG(WARNING) << "cluster exit.";

  return 0;
}
