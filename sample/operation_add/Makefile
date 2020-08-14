CXX=g++
CXXFLAGS=-std=c++11 -I../../include -Wall
TARGET=operation_add.test
HEADERS=$(shell find ../../include/aonfp -name '*.hpp')

$(TARGET):main.cpp $(HEADERS)
	$(CXX) $< -o $@ $(CXXFLAGS)

clean:
	rm -f $(TARGET)
