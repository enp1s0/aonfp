CXX=g++
CXXFLAGS=-std=c++11 -I../../
TARGET=de_compose.test

$(TARGET):main.cpp
	$(CXX) $< -o $@ $(CXXFLAGS)

clean:
	rm -f $(TARGET)
