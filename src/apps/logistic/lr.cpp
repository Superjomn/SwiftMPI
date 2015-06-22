#include "../../swiftmpi.h"
using namespace std;
using namespace swift_snails;

typedef unsigned int lr_key_t;

struct LRParam {
    float val = 0;
    float grad2sum = 0;
};
/*
struct LRLocalParam {
    float val = 0;
};
*/

typedef float LRLocalParam;

struct LRLocalGrad {
    float val = 0;
    int count = 0;

    void reset() {
        val = 0;
        count = 0;
    }
};

std::ostream& operator<< (std::ostream& os, LRParam &param) {
    os << param.val;
    return os;
}
BinaryBuffer& operator<< (BinaryBuffer &bb, LRLocalGrad &grad) {
    bb << float(grad.val / grad.count);
    return bb;
}
BinaryBuffer& operator>> (BinaryBuffer &bb, LRLocalGrad &grad) {
    bb >> grad.val;
    grad.count = 1;
    return bb;
}


class LRPullAccessMethod : public PullAccessMethod<lr_key_t, LRParam, LRLocalParam>
{
public:
    virtual void init_param(const lr_key_t &key, param_t &param) {
        param.val = global_random().gen_float();  
    }
    virtual void get_pull_value(const lr_key_t &key, const param_t &param, pull_t& val) {
        val = param.val;
    }
};


class LRPushAccessMethod : public PushAccessMethod<lr_key_t, LRParam, LRLocalGrad>
{
public:
    LRPushAccessMethod() :
        initial_learning_rate( global_config().get_config("server", "initial_learning_rate").to_float())
    { }
    /**
     * grad should be normalized before pushed
     */
    virtual void apply_push_value(const lr_key_t& key, param_t &param, const grad_t& push_val) {
        param.grad2sum += push_val.val * push_val.val;
        param.val += initial_learning_rate * push_val.val / float(std::sqrt(param.grad2sum + fudge_factor));
    }

private:
    float initial_learning_rate; 
    static const float fudge_factor;
};
const float LRPushAccessMethod::fudge_factor = 1e-6;
//LocalParamCache<lr_key_t, LRLocalParam, LRLocalGrad> param_cache;

struct Instance {
    float target;
    std::vector< std::pair<unsigned int, float>> feas;

    void clear() {
        // clear data but not free memory
        feas.clear();
    }
};

std::ostream& operator<< (std::ostream &os, const Instance &ins) {
    os << "instance:\t" << ins.target << "\t";
    for (const auto& item : ins.feas) {
        os << item.first << ":" << item.second << " ";
    }
    os << std::endl;
}

void parse_instance(const char* line, Instance &ins) {
    //RAW_LOG_INFO ("parsing:\t%s", line);
    char *cursor;
    unsigned int key;
    float value;
    CHECK((ins.target=strtod(line, &cursor), cursor != line)) << "target parse error!";
    line = cursor;
    while (*(line + count_spaces(line)) != 0) {
        CHECK((key = (unsigned int)strtoul(line, &cursor, 10), cursor != line));
        RAW_LOG_INFO ("... key:\t%d", key);
        RAW_LOG_INFO ("... value:\t%d", value);
        line = cursor;
        ins.feas.emplace_back(key, value);
    }
}

bool parse_instance2(const std::string &line, Instance &ins) {
    //RAW_LOG_INFO ("parsing \"%s\"", line.c_str());
    const char* pline = line.c_str();
    ins.feas.clear();
    while ((*pline == ' ') || (*pline == 0)) pline ++;
    if ((*pline == 0) || (*pline == '#')) return false;
    float value;
    int nchar, feature;
    if (std::sscanf(pline, "%f%n", &value, &nchar) >= 1) {
        pline += nchar;
        ins.target = value;

        while (std::sscanf(pline, "%d:%f%n", &feature, &value, &nchar) >= 2) {
            pline += nchar;
            ins.feas.emplace_back(feature, value);
        }
        //while ((*pline != 0) && ((*pline == ' ') || (*pline == 9))) pline ++;
        /*if ((*pline != 0) && (*pline != '#')) 
            throw "cannot parse line \"" + line + "\" at character " + pline[0];
        */
    } else {
        LOG(ERROR) << "parse line error";
        throw "cannot parse line \"" + line + "\" at character " + pline[0];
    }
    //LOG(INFO) << ins;
    return true;
}


class LR {
public:
    typedef GlobalPullAccess<lr_key_t, LRLocalParam, LRLocalGrad> pull_access_t;
    typedef GlobalPushAccess<lr_key_t, LRLocalParam, LRLocalGrad> push_access_t;
    typedef LocalParamCache<lr_key_t, LRLocalParam, LRLocalGrad> param_cache_t;

    LR (const string& path) : 
        _minibatch (global_config().get_config("worker", "minibatch").to_int32()),
        _nthreads (global_config().get_config("worker", "nthreads").to_int32()),
        _pull_access (global_pull_access<lr_key_t, LRLocalParam, LRLocalGrad>()),
        _push_access (global_push_access<lr_key_t, LRLocalParam, LRLocalGrad>())
    {
        _path = path; 
        CHECK_GT(_path.size(), 0);
        CHECK_GT(_minibatch, 0);
        CHECK_GT(_nthreads, 0);
        AsynExec exec(_nthreads);
        _async_channel = exec.open();
    }

    void train() {
        // init server-side parameter
        init_keys();
        LOG (WARNING) << "... first pull to init local_param_cache";
        pull();
        LOG (WARNING) << ">>> end pull()";
        global_mpi().barrier();

        std::atomic<int> line_count {0};
        LineFileReader line_reader;
        std::mutex file_mut;
        SpinLock spinlock;

        FILE* file = fopen(_path.c_str(), "rb");
        // first to init local keys
        //gather_keys(file);

        AsynExec::task_t handler = [this, 
                &line_count, &line_reader, 
                &file, &file_mut, &spinlock]() {
            std::string line;
            float error;
            char* cline;
            Instance ins;
            bool parse_res;
            while (true) {
                if (feof(file)) break;
                { std::lock_guard<std::mutex> lk(file_mut);
                    cline = line_reader.getline(file);
                    if (! cline) continue;
                    line = std::move(string(cline));
                }
                parse_res = parse_instance2(line, ins);
                if (! parse_res) continue;
                line_count ++;
                //if(ins.feas.size() < 4) continue;
                error = learn_instance(ins);
                line_count ++;
                if (line_count > _minibatch) break;
                if (feof(file)) break;
            }
        };
        LOG (WARNING) << "... to train";
        while (true) {
            line_count = 0;
            // rebuild local parameter cache
            LOG (INFO) << "... gather keys";
            gather_keys(file, _minibatch);
            _param_cache.clear();
            _param_cache.init_keys(_local_keys);
            LOG (INFO) << "... to pull minibatch";
            pull();
            LOG (INFO) << "... to multi-thread train";
            async_exec(_nthreads, handler, _async_channel);
            LOG (INFO) << "... to push minibatch";
            push();
            /*
            printf(
                "%cLines:%.2fk",
                13, float(line_count) / 1000);

            fflush(stdout);
            */
            if (feof(file)) break;
        }
        LOG(WARNING) << "finish training ...";
    }
    /**
     * @brief gather keys within a minibatch
     * @param file file with fopen
     * @param minibatch size of Mini-batch, no limit if minibatch < 0
     */
    void gather_keys(FILE* file, int minibatch = -1) {
        long cur_pos = ftell(file);
        std::atomic<int> line_count {0};
        LineFileReader line_reader;
        std::mutex file_mut;
        SpinLock spinlock;
        _local_keys.clear();
        //CounterBarrier cbarrier(_nthreads);

        AsynExec::task_t handler = [this, &line_count, &line_reader,
            &file_mut, &spinlock, minibatch, &file
        ] {
            char *cline = nullptr;
            std::string line;
            Instance ins;
            bool parse_res;
            while (true) {
                if (feof(file)) break;
                ins.clear();
                { std::lock_guard<std::mutex> lk(file_mut);
                    cline = line_reader.getline(file);
                    if (! cline) continue;
                    line = std::move(string(cline));
                }
                parse_res = parse_instance2(line, ins);
                if (! parse_res) continue;
                //if(ins.feas.size() < 4) continue;
                { std::lock_guard<SpinLock> lk(SpinLock);
                    for( const auto& item : ins.feas) {
                        _local_keys.insert(item.first);
                    }
                }
                line_count ++;
                if(minibatch > 0 &&  line_count > minibatch) break;
                if (feof(file)) break;
            }
        };
        async_exec(_nthreads, handler, _async_channel);
        RAW_LOG(INFO, "collect %d keys", _local_keys.size());
        fseek(file, cur_pos, SEEK_SET);
    }
    /**
     * SGD update
     */
    float learn_instance(const Instance &ins) {
        float sum = 0;
        for (const auto& item : ins.feas) {
            auto param = _param_cache.params()[item.first];
            sum += param * item.second;
        }
        float predict = 1. / ( 1. + exp( - sum ));
        float error = ins.target - predict;
        // update grad 
        float grad = 0;
        for (const auto& item : ins.feas) {
            grad = error * item.second;
            _param_cache.grads()[item.first].val += grad;
            _param_cache.grads()[item.first].count ++;
        }
        return error * error;
    }

protected:
    /**
     * get local keys and init local parameter cache
     */
    void init_keys() {
        string line;
        bool parse_res;
        LOG(WARNING) << "init local keys from path:\t" << _path << "...";
        ifstream file(_path);
        CHECK (file.is_open()) << "file not opened!\t" << _path;
        Instance ins;
        _local_keys.clear();
        while(getline(file, line)) {
            ins.clear();
            parse_res = parse_instance2(line, ins);
            if (! parse_res) continue;
            for (const auto& item : ins.feas) {
                _local_keys.insert(item.first);
            }
        }
        RAW_LOG_WARNING ("... to init local parameter cache");
        _param_cache.init_keys(_local_keys);
        file.close();
        RAW_LOG_WARNING (">>> finish init keys");
    }
    /**
     * query parameters contained in local cache from remote server
     */
    void pull() {
        _pull_access.pull_with_barrier(_local_keys, _param_cache);
    }
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


int main(int argc, char** argv) {
    string path = "data.txt";
    string conf = "demo.conf";
    global_config().load_conf(conf);
    global_config().parse();
    GlobalMPI::initialize(argc, argv);
    typedef ClusterServer<lr_key_t, LRParam, LRLocalParam, LRLocalGrad, LRPullAccessMethod, LRPushAccessMethod> server_t;
    Cluster<ClusterWorker, server_t, lr_key_t> cluster;
    cluster.initialize();

    LR lr(path);
    lr.train();

    cluster.finalize();
    LOG(WARNING) << "cluster exit.";

    return 0;
}
