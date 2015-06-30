Distrubted Word2Vec based on SwiftMPI
==================================================
An implementation of Word2Vec based on SwiftMPI (a tiny distributed parameter server with MPI).

Negative-Sampling + CBOW.

Compile
----------
On Unix systems, type `make` to compile `./bin/word2vec` binary file.

Input
-------
The format of training data file is:

    word1 word2 word3 word4
    word1 word2 word3 word4 word5

each line contains a space-splited setence.

You can check `data.sample.txt` for reference.

Usage
---------
To display help infomation, type:

    ./bin/word2vec
or

    ./bin/word2vec --help

To train a text file locally:

    ./bin/word2vec -config <path> -data <path> -niters <number> -output <path>

To train a text dataset with a MPI cluster, you can follow the following steps:

1. Distribute the data set to the cluster, allocate each node a file in the same path;
2. Configure MPI cluster;
3. Configure Word2Vec (check demo.conf);
4. Run Word2Vec

with command:

    MPI_command ./bin/word2vec -config <path> -data <path> -niters <number> -output <path>

`MPI_command` is MPI's command, like `mpirun -np 20` ...

After training, cluster will output word vectors to `output path` in each node.

Output Format
--------------
Word2Vec will output all model parameters to disk, the format is 

    <word-hash-code>\t<word vector>\t<word hidden vector>
    <word-hash-code>\t<word vector>\t<word hidden vector>

Configuration
--------------
Check `demo.conf` for a reference.

Several important configuration options are

* worker.minibatch: size of a minibatch, better to be larger if memory allows
* worker.nthreads: number of working threads, better to be set to the number of CPU cores.
* server.frag_num: number of parameter fragments, better to be set to several times of the number of nodes.
* word2vec.len_vec: number of dementions.
* word2vec.min_sentence_length: length of sentence requirement (or will be skipped).
* word2vec.negative: number of negative samples for each word.
