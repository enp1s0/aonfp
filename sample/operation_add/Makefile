CXX=g++
CXXFLAGS=-std=c++11 -I../../include -Wall
TARGET=operation_add.test

$(TARGET):main.cpp
	$(CXX) $< -o $@ $(CXXFLAGS)

clean:
	rm -f $(TARGET)
