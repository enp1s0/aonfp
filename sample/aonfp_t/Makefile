CXX=g++
CXXFLAGS=-std=c++17 -I../../include -Wall
HEADERS=$(shell find ../../include -name '*.hpp')
TARGET=aonfp_t.test

$(TARGET):main.cpp $(HEADERS)
	$(CXX) $< -o $@ $(CXXFLAGS)

clean:
	rm -f $(TARGET)
