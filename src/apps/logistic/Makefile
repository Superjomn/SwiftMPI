CXX=mpicxx
CC=gcc
BIN=./bin

CXXFLAGS= -g -std=c++11 -pthread -lpthread -lgtest -lgtest_main -lzmq -lglog -lz 
LOCAL_ROOT=../../../third/local

THIRD_INCPATH=-isystem $(LOCAL_ROOT)/include \
	-I/usr/include/pthread 

THIRD_LIB=-L$(LOCAL_ROOT)/lib

logistic : lr.cpp
	mkdir -p $(BIN)
	$(CXX) lr.cpp $(THIRD_INCPATH)  -Xlinker $(THIRD_LIB)  $(CXXFLAGS) -o $(BIN)/logistic
