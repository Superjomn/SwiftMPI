a Distributed Sparse Logistic Regression Implementation
========================================================
Compile
--------

    make

Dataset Format
----------------
Similar to libfm's input format:

    0 1:1 10:0.5 15:1.2
    1 2:1 9:0.4

Usage
------
    ./bin/logistic --help

    ./bin/logistic -config <config path> -dataset <dataset path>

Configuration
---------------
Read demo.conf for referernce.

Parameter Output
------------------
The final parameters will be output to ostream, use a pipe to save them.
