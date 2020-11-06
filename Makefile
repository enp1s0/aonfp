NVCC=nvcc
NVCCFLAGS=-std=c++17 -I./include
NVCCFLAGS+=-gencode arch=compute_60,code=sm_60
NVCCFLAGS+=-gencode arch=compute_61,code=sm_61
NVCCFLAGS+=-gencode arch=compute_70,code=sm_70
NVCCFLAGS+=-gencode arch=compute_75,code=sm_75
NVCCFLAGS+=-gencode arch=compute_80,code=sm_80
NVCCFLAGS+=-gencode arch=compute_86,code=sm_86

LIBDIR=lib

SRCDIR=src

OBJDIR=obj

libaonfp_cuda_copy.a: $(OBJDIR)/libaonfp_cuda_copy.o $(OBJDIR)/libaonfp_cuda_copy.dlink.o
	[ -d $(LIBDIR) ] || mkdir $(LIBDIR)
	$(NVCC) -o $(LIBDIR)/$@ $+ $(NVCCFLAGS) -lib -m64

$(OBJDIR)/libaonfp_cuda_copy.o: $(SRCDIR)/cuda_copy.cu
	[ -d $(OBJDIR) ] || mkdir $(OBJDIR)
	$(NVCC) -o $@ -c $< $(NVCCFLAGS) -dc -m64

$(OBJDIR)/libaonfp_cuda_copy.dlink.o: $(SRCDIR)/cuda_copy.cu
	[ -d $(OBJDIR) ] || mkdir $(OBJDIR)
	$(NVCC) -dlink -o $@ $< $(NVCCFLAGS) -m64

clean:
	rm -f $(OBJDIR)/*
	rm -f $(LIBDIR)/*
