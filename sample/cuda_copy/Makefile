NVCC=nvcc
NVCCFLAGS=-std=c++17 -I../../include -L../../lib -laonfp_cuda_copy
TARGET=cuda_copy.test

$(TARGET):main.cu
	$(NVCC) $< $(NVCCFLAGS) -o $@

clean:
	rm -f $(TARGET)
