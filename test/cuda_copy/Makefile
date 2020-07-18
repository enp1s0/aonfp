NVCC=nvcc
NVCCFLAGS=-std=c++11 -I../../include
TARGET=cuda_copy.test

$(TARGET):main.cu
	$(NVCC) $< $(NVCCFLAGS) -o $@

clean:
	rm -f $(TARGET)
