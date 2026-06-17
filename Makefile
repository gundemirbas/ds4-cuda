# Makefile for ds4-cuda (NVFP4 port of antirez/ds4)
# Target: DGX Spark (NVIDIA GB10, sm_121a)

CC = gcc
NVCC = nvcc
CFLAGS = -O3 -g -std=c99 -Wall -Wextra -Wno-unused-parameter -D_GNU_SOURCE -fno-finite-math-only -I.
NVFLAGS = -O3 -g --use_fast_math -gencode arch=compute_121a,code=sm_121a -Xcompiler -pthread
CUDA_LDLIBS = -lm -Xcompiler -pthread -lcudart -lcublas

# Ortak objeler (tüm binary'lerde kullanılan)
COMMON_SRCS = ds4.c ds4_help.c ds4_kvstore.c ds4_ssd.c ds4_distributed.c rax.c linenoise.c

.PHONY: all clean

all: ds4 ds4-server ds4-bench ds4-eval ds4-agent

# C compilation
%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

# CUDA compilation
%.o: %.cu
	$(NVCC) $(NVFLAGS) -I. $< -c -o $@

# CLI (ana binary)
ds4: ds4.o ds4_cli.o ds4_help.o ds4_kvstore.o ds4_ssd.o ds4_distributed.o rax.o linenoise.o ds4_cuda.o
	$(NVCC) $(NVFLAGS) $^ -o $@ $(CUDA_LDLIBS)

# Server
ds4-server: ds4.o ds4_server.o ds4_help.o ds4_kvstore.o ds4_ssd.o ds4_distributed.o rax.o linenoise.o ds4_cuda.o
	$(NVCC) $(NVFLAGS) $^ -o $@ $(CUDA_LDLIBS)

# Benchmark
ds4-bench: ds4.o ds4_bench.o ds4_help.o ds4_kvstore.o ds4_ssd.o ds4_distributed.o rax.o linenoise.o ds4_cuda.o
	$(NVCC) $(NVFLAGS) $^ -o $@ $(CUDA_LDLIBS)

# Eval
ds4-eval: ds4.o ds4_eval.o ds4_help.o ds4_kvstore.o ds4_ssd.o ds4_distributed.o rax.o linenoise.o ds4_cuda.o
	$(NVCC) $(NVFLAGS) $^ -o $@ $(CUDA_LDLIBS)

# Agent
ds4-agent: ds4.o ds4_agent.o ds4_web.o ds4_help.o ds4_kvstore.o ds4_ssd.o ds4_distributed.o rax.o linenoise.o ds4_cuda.o
	$(NVCC) $(NVFLAGS) $^ -o $@ $(CUDA_LDLIBS)

clean:
	rm -f *.o ds4 ds4-server ds4-bench ds4-eval ds4-agent
