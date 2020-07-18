NVCC=nvcc
NVCCFLAGS=-std=c++11 -arch=sm_70 -I../
TARGET=aonfp-sample.out
HEADERS=$(shell find ../aonfp -name '*.hpp')

$(TARGET):main.cu $(HEADERS)
	$(NVCC) $< -o $@ $(NVCCFLAGS)

clean:
	rm -f $(TARGET)
