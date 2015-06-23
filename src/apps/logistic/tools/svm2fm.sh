#!/bin/bash

awk -F" " '{
    if($1 == "+1") printf 1 " ";
    else if($1 == "-1") printf 0 " ";
    for(i = 2; i < NF; i++) {
        printf $i " ";
    }
    printf $NF "\n";
}' $1
