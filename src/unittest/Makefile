CXX=mpicxx
CC=gcc
BIN=./bin

CXXFLAGS= -g -std=c++11 -pthread -lpthread -Wall -lgtest -lgtest_main -lzmq -lglog  -lz
LOCAL_ROOT=../../third/local
GTEST_HOME=$(LOCAL_ROOT)/gtest

THIRD_INCPATH=-I$(LOCAL_ROOT)/glog/include \
	-I$(LOCAL_ROOT)/zeromq/include \
	-I$(LOCAL_ROOT) \
	-I/usr/local/Cellar/google-sparsehash/2.0.2/include \
	-I/usr/include/pthread \
	-I$(GTEST_HOME)/include

THIRD_LIB=-L$(LOCAL_ROOT)/glog/lib/  -L$(LOCAL_ROOT)/zeromq/lib/ -L$(LOCAL_ROOT)/zlib/lib/

test :  main.cpp 
	mkdir -p $(BIN)
	$(CXX) main.cpp $(THIRD_INCPATH) -L$(GTEST_HOME)/lib/.libs -Xlinker $(THIRD_LIB)  $(CXXFLAGS) -o $(BIN)/test

mpi_test: ./mpi_test.cpp ../utils/mpi.h
	mkdir -p $(BIN)
	mpicxx mpi_test.cpp $(THIRD_INCPATH) -L$(GTEST_HOME)/lib/.libs -Xlinker $(THIRD_LIB)  $(CXXFLAGS) -o $(BIN)/mpi_test


