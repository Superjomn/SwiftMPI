//
//  ConfigParser.h
//  SwiftSnails
//
//  Created by Chunwei on 12/2/14.
//  Copyright (c) 2014 Chunwei. All rights reserved.
//

#ifndef SwiftSnails_utils_ConfigParser_h_
#define SwiftSnails_utils_ConfigParser_h_
#include "common.h"
#include "string.h"

namespace swift_snails {
/**
 * a simple config parser
 *
 * support syntax:
 *
 *   import ../common.conf
 *   key: value
 *   # comment
 *   key: value
 */
class ConfigParser : VirtualObject {

public:
    struct Item {
        std::string value;
        Item(const std::string &value) : value(value){ }
        int to_int32() const {
            CHECK(!value.empty());
            return std::move(stoi(value));
        }
        float to_float() const {
            CHECK(!value.empty());
            return std::move(stof(value));
        }
        std::string to_string() const {
            return value;
        }
        bool to_bool() const {
            CHECK(!value.empty());
            CHECK(value == "true" || value == "false");
            if(value == "true") return true;
            return false;
        }
    };
    
    explicit ConfigParser(const std::string &conf_path): _conf_path(conf_path) {
    }

    explicit ConfigParser() {
    }
    void load_conf(const std::string path) {
        _conf_path = path;
    }

    void parse() {
        LOG(WARNING) << "load conf from\t" << _conf_path;
        parse_conf(_conf_path);
    }

    void clear() {
        _session.clear();
    }
    // to_int32()
    // to_string()
    // to_bool()
    const Item& get_config(const std::string &session, const std::string &key) {
        auto &s = _session[session];
        auto p = s.find(key);
        CHECK(p != s.end()) << "no such key:\t[" << session << "]\t" << key;
        return p->second;
    }

    friend std::ostream& operator<< (std::ostream& os, const ConfigParser &other) {
        os << "conf:" << std::endl;
        for(auto& session : other._session) {
            os << "\tsession:\t" << session.first << std::endl;
            for(auto kv : session.second) {
                os << "\t\t" << kv.first << "\t" << kv.second.value << std::endl;
            }
        }
        os << "end conf" << std::endl;
        return os;
    }

private:

    void parse_conf(const std::string &path) {
        //LOG(INFO) << "parsing conf:\t" << path;
        typedef std::pair<std::string, std::string> key_value_t;
        std::ifstream file(path);
        PCHECK(file) << "conf can not open:\t" << path;
        std::string line;
        while(getline(file, line)) {
            trim(line);
            if(line.empty()) continue;
            if(headswith(line, "#")) continue;  // skip comments
            // import path
            if(headswith(line, "import")) {
                std::pair<std::string, std::string> kv = key_value_split(line, " ");
                CHECK(kv.second != path);   // 
                parse_conf(kv.second);
                continue;
            }
            // parse session
            if(line[0] == '[' && line[line.length()-1] == ']') {
                cur_session = line.substr(1, line.size() - 2);
                LOG(INFO) << "parsing session" << cur_session;
                trim(cur_session);
                CHECK(! cur_session.empty());
                continue;
            }
            key_value_t kv = std::move(key_value_split(line, ":"));
            set_config(trim(kv.first), trim(kv.second));
        }
        file.close();
    }

    void set_config(const std::string &key, const std::string &value) {
        if(_session.count(cur_session) == 0) {
            _session.emplace(cur_session, 
                std::map<std::string, Item>());
        }
        _session[cur_session].insert({key, Item(value)});
    }

    std::map<std::string, std::map<std::string, Item>> _session;
    std::string _conf_path;
    std::string cur_session;
};  // end class ConfigParser


// global_config need to be inited
ConfigParser& global_config() {
    static ConfigParser config;
    return config;
}

/*
void init_configs(const std::string& configs) {
    std::vector<std::string> _configs = split(configs, " \n");

    for (auto& c : _configs) {
        global_config().register_config(c);
    }
}
*/

}; // end namespace swift_snails

#endif
