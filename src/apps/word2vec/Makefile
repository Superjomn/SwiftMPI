CXX=mpicxx
CC=gcc
BIN=./bin

CXXFLAGS= -g -std=c++11 -pthread -lpthread -lgtest -lgtest_main -lzmq -lz -lgflags -lglog
LOCAL_ROOT=../../../third/local

THIRD_INCPATH=-isystem $(LOCAL_ROOT)/include \
	-I/usr/include/pthread  \
	-I/usr/local/Cellar/google-sparsehash/2.0.2/include 

THIRD_LIB=-L$(LOCAL_ROOT)/lib  

word2vec : w2v.cpp
	mkdir -p $(BIN)
	$(CXX) w2v.cpp $(THIRD_INCPATH)  -Xlinker $(THIRD_LIB)  $(CXXFLAGS) -o $(BIN)/word2vec
