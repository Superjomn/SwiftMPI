//
//  file.h
//  SwiftSnails
//
//  Created by Chunwei on 3/28/15.
//  Copyright (c) 2015 Chunwei. All rights reserved.
//
#pragma once
#include "common.h"
#include "string.h"

namespace swift_snails {
// thread-safe
void scan_file_by_line (
        FILE *file, 
        std::mutex& file_mut, 
        std::function<void(const std::string &line)> &&handler
    )
{
    LineFileReader line_reader;
    for(;;) {

        file_mut.lock();
        char* res = line_reader.getline(file);

        if(res != NULL) {
            std::string line = line_reader.get();
            file_mut.unlock();
            // gather keys
            handler(line);

        } else {
            file_mut.unlock();
            return;
        }
    }
}

/**
 * parse file with keys like:
 *  112 113 224 445
 */
std::vector<size_t> parse_keys_file(const std::string &line, const std::string &spliter = " ") {
    std::vector<size_t> res;
    auto fields = std::move(split(line, spliter));
    for (std::string & f : fields) {
        res.push_back(std::stoi(f));
    }
	return std::move(res);
}
