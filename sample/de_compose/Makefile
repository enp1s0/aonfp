CXX=g++
CXXFLAGS=-std=c++11 -I../../include -Wall
TARGET=de_compose.test

$(TARGET):main.cpp
	$(CXX) $< -o $@ $(CXXFLAGS)

clean:
	rm -f $(TARGET)
