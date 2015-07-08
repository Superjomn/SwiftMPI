#include "word2vec.h"
//#include "word2vec_global.h"
//#include <gflags/gflags.h>
using namespace std;
int main(int argc, char* argv[]) {
    GlobalMPI::initialize(argc, argv);
    // init config
    fms::CMDLine cmdline(argc, argv);
    std::string param_help         = cmdline.registerParameter("help",   "this screen");
    std::string param_config_path  = cmdline.registerParameter("config", "path of config file          \t[string]");
    std::string param_data_path    = cmdline.registerParameter("data",   "path of dataset, text only!  \t[string]");
    std::string param_niters       = cmdline.registerParameter("niters", "number of iterations         \t[int]");
    std::string param_param_output = cmdline.registerParameter("output", "path to output the parameters\t[string]");

    if(cmdline.hasParameter(param_help) || argc == 1) {
        cout << endl;
        cout << "===================================================================" << endl;
        cout << "   Word2Vec application" << endl;
        cout << "   Author: Suprjom <yanchunwei@outlook.com>" << endl;
        cout << "===================================================================" << endl;
        cmdline.print_help();
        cout << endl;
        cout << endl;
        return 0;
    }
    if (!cmdline.hasParameter(param_config_path) ||
        !cmdline.hasParameter(param_data_path) ||
        !cmdline.hasParameter(param_niters)
    ) {
        LOG(ERROR) << "missing parameter";
        cmdline.print_help();
        return 0;
    }
    std::string config_path = cmdline.getValue(param_config_path);
    std::string data_path   = cmdline.getValue(param_data_path);
    std::string output_path = cmdline.getValue(param_param_output);
    int niters              = stoi(cmdline.getValue(param_niters));
    global_config().load_conf(config_path);
    global_config().parse();

    // init cluster
    Cluster<ClusterWorker, server_t, w2v_key_t> cluster;
    cluster.initialize();

    Word2Vec<MiniBatch> w2v(data_path, niters);
    w2v.train();
    swift_snails::format_string(output_path, "-%d.txt", global_mpi().rank());
    RAW_LOG_WARNING ("server output parameter to %s", output_path.c_str());
    cluster.finalize(output_path);

    LOG(WARNING) << "cluster exit.";

    return 0;
}
