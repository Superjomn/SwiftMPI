#include "../../swiftmpi.h"
#include "../word2vec/word2vec.h"

class SentMiniBatch : public MiniBatch {
public:
    virtual void clear() noexcept {
        _local_keys.clear();
        _word_freq.clear();
        _wordids.clear();
        _sentids.clear();
    }
    /**
     * @brief gather wordids and sentids
     * @warning _local_keys = wordids + sentids
     */
    size_t gather_keys (FILE* file, int minibatch = 0) noexcept {
        long cur_pos = ftell(file);
        std::atomic<int> line_count {0};
        LineFileReader line_reader;
        std::mutex file_mut;
        SpinLock spinlock1, spinlock2;
        _local_keys.clear();
        AsynExec::task_t handler = [this, &line_count, &line_reader,
            &file_mut, &spinlock1, &spinlock2, minibatch, &file
        ] {
            char *cline = nullptr;
            std::string line;
            Instance ins;
            std::map<w2v_key_t, int>::iterator word_freq_it;
            bool parse_res;
            while (true) {
                if (feof(file)) break;
                ins.clear();
                { std::lock_guard<std::mutex> lk(file_mut);
                    cline = line_reader.getline(file);
                    if (! cline) continue;
                    line = std::move(std::string(cline));
                }
                // get sentence id
                ins.sent_id = hash_fn(line.c_str());
                _sentids.insert(ins.sent_id);
                // treat sent_id as special word_id
                _local_keys.insert(ins.sent_id);
                // gent word ids
                parse_res = parse_instance(line, ins);
                if (! parse_res) continue;
                for( const auto& item : ins.words) {
                    { std::lock_guard<SpinLock> lk(spinlock1);
                        _local_keys.insert(item);
                    }
                    { std::lock_guard<SpinLock> lk(spinlock2);
                        word_freq_it = _word_freq.find(item);
                        if (word_freq_it != _word_freq.end()) 
                            word_freq_it->second ++;
                        else 
                            _word_freq[item] = 1;
                    }
                }
                line_count ++;
                if(minibatch > 0 &&  line_count > minibatch) break;
                if (feof(file)) break;
            }
        };
        async_exec(_nthreads, handler, global_channel());
        RAW_LOG(INFO, "collect %d keys", _local_keys.size());
        fseek(file, cur_pos, SEEK_SET);
        return _word_freq.size();
    }
    /**
     * just push sentence vector's gradient
     */
    void push() noexcept {
        _push_access.push_with_barrier(_sentids, _param_cache);
        clear();
    }

protected:
    std::unordered_set<w2v_key_t> _sentids;
};


class Doc2Vec : public Word2Vec<SentMiniBatch> {
public:
    Doc2Vec (const std::string& path, int niters) :
        Word2Vec (path, niters)
    { }

    void load_param (const std::string &path) {
        global_server<server_t>().load(path);
        global_mpi().barrier();
    }

protected:
    void learn_instance (Instance &ins, Vec& neu1, Vec& neu1e) noexcept {
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

            neu1 = _minibatch.param().params()[ins.sent_id].v;

            for (a = b; a < _window * 2 + 1 - b; a++) {
                if (a != _window) {
                    c = pos - _window + a;
                    if (c < 0 || c >= sent_length) continue;
                    last_word = ins.words[c];
                    Vec& syn0_lastword = _minibatch.param().params()[last_word].v;
                    neu1 += syn0_lastword;
                }
            }
            for (int d = 0; d < _negative + 1; d++) {
                if (d == 0) {
                    target = word;
                    label = 1;
                // generate negative samples
                } else {
                    target = _minibatch.table()[(global_random()() >> 16) % table_size];
                    if (target == 0) 
                        target = _minibatch.table()[(global_random()() >> 16) % table_size];
                    if (target == word) continue;
                    label = 0;
                }
                Vec& syn1neg_target = _minibatch.param().params()[target].h;
                f = 0;
                f += neu1.dot(syn1neg_target);
                if (f > MAX_EXP) g = (label - 1) * _alpha;
                else if (f < -MAX_EXP) g = (label - 0) * _alpha;
                else g = (label - exptable(f)) * _alpha;
                _error.accu(10000 * g * g);
                neu1e += g * syn1neg_target;
                _minibatch.param().grads()[target].accu_h(g * neu1);
            }
            // update sentence vector
            _minibatch.param().grads()[ins.sent_id].accu_v(neu1e);
            /*
            for (a = b; a < _window * 2 + 1 - b; a++) {
                if (a != _window) {
                    c = pos - _window + a;
                    if (c < 0 || c >= sent_length) continue;
                    last_word = ins.words[c];
                    Vec &syn0_lastword = _minibatch.param().params()[last_word].v;
                    _minibatch.param().grads()[last_word].accu_v( neu1e);
                }
            }
            */
        }
    }

private:
    // dataset path
    std::string _path;
    int _batchsize; 
    int _nthreads;  
    int _niters;
    int _window;
    int _negative;
    std::unordered_set<w2v_key_t> _local_keys;
    float _alpha;   // learning rate
    MiniBatch _minibatch;
    Error _error;
};  // end class Word2Vec

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

    Doc2Vec d2v(data_path, niters);
    LOG (WARNING) << "... loading word vectors";
    d2v.load_param(word_vec_path);
    d2v.train();
    swift_snails::format_string(output_path, "-%d.txt", global_mpi().rank());
    RAW_LOG_WARNING ("server output parameter to %s", output_path.c_str());
    cluster.finalize(output_path);

    LOG(WARNING) << "cluster exit.";

    return 0;
}
