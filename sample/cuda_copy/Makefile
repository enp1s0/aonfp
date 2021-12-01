NVCC=nvcc
NVCCFLAGS=-std=c++17 -I../../include -L../../build/ -laonfp_cuda_copy
NVCCFLAGS+=-gencode arch=compute_60,code=sm_60
NVCCFLAGS+=-gencode arch=compute_61,code=sm_61
NVCCFLAGS+=-gencode arch=compute_70,code=sm_70
NVCCFLAGS+=-gencode arch=compute_75,code=sm_75
NVCCFLAGS+=-gencode arch=compute_80,code=sm_80
NVCCFLAGS+=-gencode arch=compute_86,code=sm_86
TARGET=cuda_copy.test

$(TARGET):main.cu
	$(NVCC) $< $(NVCCFLAGS) -o $@

clean:
	rm -f $(TARGET)
