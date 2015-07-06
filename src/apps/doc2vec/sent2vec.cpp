#include <functional>
#include "../../swiftmpi.h"
#include "../word2vec/word2vec_global.h"
using namespace swift_snails;


class WordMiniBatch : public MiniBatch {
public:
    WordMiniBatch() 
    { }
    WordMiniBatch(param_cache_t *param) :
        MiniBatch(param)
    { }

    void push() = delete;
    //size_t gather_keys (FILE* file, int &line_id, int minibatch, int nthreads=0) = delete;
};


class Sent2Vec : public VirtualObject {
public:
    Sent2Vec(const std::string& path, const std::string& out_path, int niters) :
        _path (path),
        _out_path (out_path), 
        _batchsize (global_config().get("worker", "minibatch").to_int32()),
        _nthreads (global_config().get("worker", "nthreads").to_int32()),
        _window (global_config().get("word2vec", "window").to_int32()),
        _negative (global_config().get("word2vec", "negative").to_int32()),
        _alpha (global_config().get("word2vec", "learning_rate").to_float()),
        _niters (niters)
    {
        _path = path; 
        CHECK_GT(_path.size(), 0);
        CHECK_GT(_batchsize, 0);
        CHECK_GT(_nthreads, 0);
        CHECK_GT(_niters, 0);
        _word_minibatch.init_param(&_word_param_cache);
    }

    void load_word_vector (const std::string &path) {
        global_server<server_t>().load(path);
        global_mpi().barrier();
    }

    void train() {
        LOG (INFO) << "first pull to init local word paramters";
        FILE* file = fopen(_path.c_str(), "rb");
        std::ofstream ofile(_out_path.c_str());
        nlines = 0;
        if (_word_minibatch.gather_keys(file, nlines) < 5) return;
        LOG (INFO) << "local data has " << nlines << " lines\t" << train_words << " words";
        LOG (INFO) << "to pull ...";
        _word_minibatch.pull();
        _word_minibatch.param().grads().clear();
        CHECK (_word_minibatch.param().grads().empty()) << "grads is not needed";
        global_mpi().barrier();
        // TODO begin to learn sentence vector
        std::mutex file_mut;
        std::mutex out_file_mut;
        std::atomic<int> line_count{0};
        int line_id {0}, _line_id;
        LineFileReader line_reader(file);
        SpinLock spinlock;
        Instance ins;

        AsynExec::task_t handler = [this,
            &file, &file_mut, &ofile, &out_file_mut, &line_reader, &line_id, 
            &line_count, &spinlock
            ] {
                std::string line;
                char* cline;
                Instance ins;
                bool parse_res;
                Vec neu1(len_vec()), neu1e(len_vec());
                Vec sent_vec(len_vec());
                float error;
                while (true) {
                    if (feof(file)) break;
                    { std::lock_guard<std::mutex> lk(file_mut);
                        cline = line_reader.getline();
                        if (! cline) continue;
                        line = std::move(std::string(cline));
                    }
                    parse_res = parse_instance(line, ins);
                    if (! parse_res) continue;

                    w2v_key_t sent_id = hash_fn(line.c_str());
                    sent_vec.random();  // init

                    for (int i = 0; i < _niters; i++) 
                        error = learn_instance(ins, neu1, neu1e, sent_vec);
                    _error.accu(error);

                    { std::lock_guard<std::mutex> lk(out_file_mut);
                        ofile << sent_id << "\t" << sent_vec << std::endl;
                    }
                    if (++ line_id % _batchsize == 0) 
                        RAW_LOG_INFO( "train status:\t%f\t%d/%d", (float)line_id/nlines, line_id, nlines);
                    if (feof(file)) break;
                }
        };

        async_exec(_nthreads, handler, global_channel());

        LOG (WARNING) << "error:\t" << _error.norm();
    }

protected:
    float learn_instance (Instance &ins, Vec& neu1, Vec& neu1e, Vec& sent_vec) noexcept {
        //neu1.clear(); neu1e.clear();
        int a, c, b = global_random()() % _window;
        int sent_length = ins.words.size();
        int pos = 0;
        int label;
        float g, f;
        w2v_key_t word, target, last_word;

        for (pos = 0; pos < sent_length; pos ++) {
            word = ins.words[pos];
            neu1.clear(); neu1e.clear();
            b = global_random()() % _window;

            neu1 = sent_vec;
            for (a = b; a < _window * 2 + 1 - b; a++) {
                if (a != _window) {
                    c = pos - _window + a;
                    if (c < 0 || c >= sent_length) continue;
                    last_word = ins.words[c];
                    Vec& syn0_lastword = _word_minibatch.param().params()[last_word].v;
                    neu1 += syn0_lastword;
                }
            }
            for (int d = 0; d < _negative + 1; d++) {
                if (d == 0) {
                    target = word;
                    label = 1;
                // generate negative samples
                } else {
                    target = _word_minibatch.table()[(global_random()() >> 16) % table_size];
                    if (target == 0) 
                        target = _word_minibatch.table()[(global_random()() >> 16) % table_size];
                    if (target == word) continue;
                    label = 0;
                }
                Vec& syn1neg_target = _word_minibatch.param().params()[target].h;
                f = 0;
                f += neu1.dot(syn1neg_target);
                if (f > MAX_EXP) g = (label - 1) * _alpha;
                else if (f < -MAX_EXP) g = (label - 0) * _alpha;
                else g = (label - exptable(f)) * _alpha;
                //_error.accu(10000 * g * g);
                neu1e += g * syn1neg_target;
                //_word_minibatch.param().grads()[target].accu_h(g * neu1);
            }
            // hidden -> in
            for (a = b; a < _window * 2 + 1 - b; a++) {
                if (a != _window) {
                    c = pos - _window + a;
                    if (c < 0 || c >= sent_length) continue;
                    last_word = ins.words[c];
                    Vec &syn0_lastword = _word_minibatch.param().params()[last_word].v;
                    //_word_minibatch.param().grads()[last_word].accu_v( neu1e);
                    sent_vec += _alpha * neu1e;
                }
            }
        }
        return g * g;
    }

private:
    // dataset path
    std::string _path;
    std::string _out_path;
    int _nthreads;  
    float _alpha;
    int _batchsize;
    int _niters;
    int nlines;
    int _window;
    int _negative;
    WordMiniBatch           _word_minibatch;
    MiniBatch::param_cache_t _word_param_cache;
    Error _error;
};


using namespace std;
int main(int argc, char* argv[]) {
    GlobalMPI::initialize(argc, argv);
    // init config
    fms::CMDLine cmdline(argc, argv);
    std::string param_help         = cmdline.registerParameter("help",   "this screen");
    std::string param_config_path  = cmdline.registerParameter("config", "path of config file          \t[string]");
    std::string param_word_vec     = cmdline.registerParameter("wordvec","path of word vector          \t[string]");
    std::string param_data_path    = cmdline.registerParameter("data",   "path of dataset, text only!  \t[string]");
    std::string param_niters       = cmdline.registerParameter("niters", "number of iterations         \t[int]");
    std::string param_param_output = cmdline.registerParameter("output", "path to output paragraph vectors\t[string]");

    if(cmdline.hasParameter(param_help) || argc == 1) {
        cout << endl;
        cout << "===================================================================" << endl;
        cout << "   Doc2Vec (Distributed Paragraph Vector)" << endl;
        cout << "   Author: Suprjom <yanchunwei@outlook.com>" << endl;
        cout << "===================================================================" << endl;
        cmdline.print_help();
        cout << endl;
        cout << endl;
        return 0;
    }
    if (!cmdline.hasParameter(param_config_path) ||
        !cmdline.hasParameter(param_data_path) ||
        !cmdline.hasParameter(param_word_vec) ||
        !cmdline.hasParameter(param_niters)
    ) {
        LOG(ERROR) << "missing parameter";
        cmdline.print_help();
        return 0;
    }
    std::string config_path = cmdline.getValue(param_config_path);
    std::string data_path   = cmdline.getValue(param_data_path);
    std::string output_path = cmdline.getValue(param_param_output);
    std::string word_vec_path = cmdline.getValue(param_word_vec);
    int niters              = stoi(cmdline.getValue(param_niters));
    global_config().load_conf(config_path);
    global_config().parse();

    // init cluster
    Cluster<ClusterWorker, server_t, w2v_key_t> cluster;
    cluster.initialize();

    Sent2Vec sent2vec(data_path, output_path, niters);
    LOG (WARNING) << "... loading word vectors";
    sent2vec.load_word_vector(word_vec_path);
    LOG (WARNING) << "... to train";
    sent2vec.train();

    LOG(WARNING) << "cluster exit.";
    return 0;
}
