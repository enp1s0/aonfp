CXX=g++
CXXFLAGS=-std=c++11 -I../../include -Wall
HEADERS=$(shell find ../../include -name '*.hpp')
TARGET=utils.test

$(TARGET):main.cpp $(HEADERS)
	$(CXX) $< -o $@ $(CXXFLAGS)

clean:
	rm -f $(TARGET)
