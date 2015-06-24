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
    ./bin/logistic --helpshort

Train Mode:

    ./bin/logistic -mode 0 -config <config path> -dataset <dataset path> -niters <number of iterations>

Predict Mode:

    ./bin/logistic -mode 1 -config <config path> -dataset <dataset path> -param_path <path of parameter> -out_prefix <path to output predictions>

Configuration
---------------
Read demo.conf for referernce.

Parameter Output
------------------
The final parameters will be output to ostream, use a pipe to save them.
