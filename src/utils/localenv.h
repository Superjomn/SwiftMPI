//
//  localenv.h
//  core
//
//  Created by Chunwei on 2/6/15.
//  Copyright (c) 2015 Chunwei. All rights reserved.
//
#pragma once
#include <sstream>
#include <stdio.h>
#include <unistd.h>
#include <string.h> /* for strncpy */
#include <sys/types.h>
#include <sys/socket.h>
#include <ifaddrs.h>
#include <netinet/in.h> 
#include <string.h> 
#include <arpa/inet.h>
#include "../utils/common.h"
#include "../utils/string.h"
#include "../utils/Buffer.h"
namespace swift_snails {

std::string get_local_ip() {
    struct ifaddrs * ifAddrStruct=NULL;
    struct ifaddrs * ifa=NULL;
    void * tmpAddrPtr=NULL;
    std::string ip;

    getifaddrs(&ifAddrStruct);

    for (ifa = ifAddrStruct; ifa != NULL; ifa = ifa->ifa_next) {
        if (!ifa->ifa_addr) {
            continue;
        }
        if (ifa->ifa_addr->sa_family == AF_INET) { // check it is IP4
            // is a valid IP4 Address
            tmpAddrPtr=&((struct sockaddr_in *)ifa->ifa_addr)->sin_addr;
            char addressBuffer[INET_ADDRSTRLEN];
            inet_ntop(AF_INET, tmpAddrPtr, addressBuffer, INET_ADDRSTRLEN);
            if(strcmp(addressBuffer, "127.0.0.1") != 0) {
                ip = addressBuffer;
            }
            //printf("%s IP Address %s\n", ifa->ifa_name, addressBuffer); 
        } 
    }
    if (ifAddrStruct!=NULL) freeifaddrs(ifAddrStruct);
    return std::move(ip);
}




};  // end namespace swift_snails
