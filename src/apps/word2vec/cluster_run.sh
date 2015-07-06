#../../../third/local/bin/mpirun -np 5 
mpirun  -np 3  -map-by node -hostfile hosts -cpu-set 0-15 --debug-devel ./bin/word2vec -config demo.conf -data ~/data.sample -niters 1 -output 1.param 
#./bin/word2vec -config demo.conf -data ~/data.sample.txt -niters 1 -output 1.param
