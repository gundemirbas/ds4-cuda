# Makefile for ds4-cuda (NVFP4 port of antirez/ds4)
# Target: DGX Spark (NVIDIA GB10, sm_121a)
#
# All source files are merged into ds4.c (CPU) and ds4_cuda.cu (CUDA).
# Separate files from AGENTS.md have been merged and are no longer compiled.

CC = gcc
NVCC = nvcc
CFLAGS = -O3 -g -std=c99 -Wall -Wextra -Wno-unused-parameter -D_GNU_SOURCE -fno-finite-math-only -I.
NVFLAGS = -O3 -g --use_fast_math -gencode arch=compute_121a,code=sm_121a -Xcompiler -pthread
CUDA_LDLIBS = -lm -Xcompiler -pthread -lcudart -lcublas

.PHONY: all clean

all: ds4 ds4-server ds4-bench ds4-eval ds4-agent

# C compilation
%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

# CUDA compilation
%.o: %.cu
	$(NVCC) $(NVFLAGS) -I. $< -c -o $@

# Ortak objeler (merged: ds4_safetensors.c, ds4_kv_cache.c, ds4_model_config.c, ds4_expert_cache.c)
COMMON = ds4.o ds4_help.o ds4_kvstore.o ds4_ssd.o ds4_distributed.o rax.o linenoise.o
CUDA = ds4_cuda.o

# CLI
ds4: $(COMMON) ds4_cli.o $(CUDA)
	$(NVCC) $(NVFLAGS) $^ -o $@ $(CUDA_LDLIBS)

# Server
ds4-server: $(COMMON) ds4_server.o $(CUDA)
	$(NVCC) $(NVFLAGS) $^ -o $@ $(CUDA_LDLIBS)

# Benchmark
ds4-bench: $(COMMON) ds4_bench.o $(CUDA)
	$(NVCC) $(NVFLAGS) $^ -o $@ $(CUDA_LDLIBS)

# Eval
ds4-eval: $(COMMON) ds4_eval.o $(CUDA)
	$(NVCC) $(NVFLAGS) $^ -o $@ $(CUDA_LDLIBS)

# Agent
ds4-agent: $(COMMON) ds4_agent.o ds4_web.o $(CUDA)
	$(NVCC) $(NVFLAGS) $^ -o $@ $(CUDA_LDLIBS)

clean:
	rm -f *.o ds4 ds4-server ds4-bench ds4-eval ds4-agent
