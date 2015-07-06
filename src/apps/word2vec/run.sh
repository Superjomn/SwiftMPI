#../../../third/local/bin/mpirun -np 5 
#../../../third/local/bin/mpirun -np 1  -cpu-set 0-15 ./bin/word2vec -config demo.conf -data ~/data.sample.txt -niters 10 -output 1.param 
./bin/word2vec -config demo.conf -data ./data.sample.txt -niters 10 -output 1.param 
#./bin/word2vec -config demo.conf -data ~/data.sample.txt -niters 1 -output 1.param
