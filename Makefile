# ds4-cuda Makefile — NVFP4/MXFP4 Tensor Core Inference for DGX Spark
# Targets: sm_121 (DGX Spark / NVIDIA GB10)

CC       = gcc
CXX      = g++
NVCC     ?= /usr/local/cuda-13.0/bin/nvcc
CFLAGS   = -O3 -g -std=c99 -Wall -Wextra -Wno-unused-parameter
CXXFLAGS = -O3 -g -std=c++17 -Wall -Wextra
NVFLAGS  = -O3 --use_fast_math -gencode arch=compute_121a,code=sm_121a

INCDIR  = include
SRCDIR  = src
CUDA_DIR = cuda
BUILDDIR = build

LIBS = -lm -lcudart -lcublas

C_OBJS = \
    $(BUILDDIR)/ds4_safetensors.o \
    $(BUILDDIR)/ds4_expert_cache.o \
    $(BUILDDIR)/ds4_model_config.o \
    $(BUILDDIR)/ds4_kv_cache.o

CUDA_OBJS = \
    $(BUILDDIR)/ds4_cuda_nvfp4_mmq.o \
    $(BUILDDIR)/ds4_cuda_forward.o \
    $(BUILDDIR)/ds4_cuda_embedding.o \
    $(BUILDDIR)/ds4_cuda_fp8_attention.o \
    $(BUILDDIR)/ds4_kv_cache_cu.o \
    $(BUILDDIR)/ds4_main.o \
    $(BUILDDIR)/ds4_layer_forward.o

ALL_OBJS = $(C_OBJS) $(CUDA_OBJS)

.PHONY: all clean test

all: $(BUILDDIR)/ds4

$(BUILDDIR)/ds4_safetensors.o: $(SRCDIR)/ds4_safetensors.c | $(BUILDDIR)
	$(CC) $(CFLAGS) -I$(INCDIR) -c $< -o $@

$(BUILDDIR)/ds4_expert_cache.o: $(SRCDIR)/ds4_expert_cache.c | $(BUILDDIR)
	$(CC) $(CFLAGS) -I$(INCDIR) -c $< -o $@

$(BUILDDIR)/ds4_model_config.o: $(SRCDIR)/ds4_model_config.c | $(BUILDDIR)
	$(CC) $(CFLAGS) -I$(INCDIR) -c $< -o $@

$(BUILDDIR)/ds4_kv_cache.o: $(SRCDIR)/ds4_kv_cache.c | $(BUILDDIR)
	$(CC) $(CFLAGS) -I$(INCDIR) -c $< -o $@

$(BUILDDIR)/ds4_cuda_nvfp4_mmq.o: $(CUDA_DIR)/ds4_cuda_nvfp4_mmq.cu | $(BUILDDIR)
	$(NVCC) $(NVFLAGS) -I$(INCDIR) -I$(CUDA_DIR) -c $< -o $@

$(BUILDDIR)/ds4_cuda_forward.o: $(CUDA_DIR)/ds4_cuda_forward.cu | $(BUILDDIR)
	$(NVCC) $(NVFLAGS) -I$(INCDIR) -I$(CUDA_DIR) -c $< -o $@

$(BUILDDIR)/ds4_cuda_embedding.o: $(CUDA_DIR)/ds4_cuda_embedding.cu | $(BUILDDIR)
	$(NVCC) $(NVFLAGS) -I$(INCDIR) -I$(CUDA_DIR) -c $< -o $@

$(BUILDDIR)/ds4_cuda_fp8_attention.o: $(CUDA_DIR)/ds4_cuda_fp8_attention.cu | $(BUILDDIR)
	$(NVCC) $(NVFLAGS) -I$(INCDIR) -I$(CUDA_DIR) -c $< -o $@

$(BUILDDIR)/ds4_kv_cache_cu.o: $(SRCDIR)/ds4_kv_cache.cu | $(BUILDDIR)
	$(NVCC) $(NVFLAGS) -I$(INCDIR) -I$(CUDA_DIR) -c $< -o $@

$(BUILDDIR)/ds4_main.o: $(SRCDIR)/ds4_main.cu | $(BUILDDIR)
	$(NVCC) $(NVFLAGS) -I$(INCDIR) -I$(CUDA_DIR) -c $< -o $@

$(BUILDDIR)/ds4_layer_forward.o: $(SRCDIR)/ds4_layer_forward.cu | $(BUILDDIR)
	$(NVCC) $(NVFLAGS) -I$(INCDIR) -I$(CUDA_DIR) -c $< -o $@

$(BUILDDIR)/ds4: $(ALL_OBJS)
	$(NVCC) $(NVFLAGS) -I$(INCDIR) -I$(CUDA_DIR) $^ -o $@ $(LIBS)

$(BUILDDIR):
	mkdir -p $(BUILDDIR)

test: $(BUILDDIR)/test_safetensors
	./$(BUILDDIR)/test_safetensors

$(BUILDDIR)/test_safetensors: $(BUILDDIR)/ds4_safetensors.o $(SRCDIR)/ds4_safetensors.c
	$(CC) $(CFLAGS) -I$(INCDIR) $^ -o $@ -lm

clean:
	rm -rf $(BUILDDIR)
