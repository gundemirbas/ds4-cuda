/**
 * ds4_kv_cache.cu — FP8 KV Cache CUDA implementation
 */

#include "ds4_kv_cache.h"
#include <cuda_runtime.h>
#include <stdio.h>

#define CUDA_CHECK(call) do { \
    cudaError_t err = (call); \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, \
                cudaGetErrorString(err)); \
        return NULL; \
    } \
} while(0)

/* ========================================================================
 * CUDA Kernels
 * ======================================================================== */

__global__ void fp32_to_fp8_kernel(const float *in, uint8_t *out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float v = in[i];
    uint32_t bits;
    memcpy(&bits, &v, sizeof(uint32_t));
    uint32_t sign = (bits >> 31) & 0x80;
    int32_t exp = ((bits >> 23) & 0xFF) - 127 + 7;
    uint32_t mantissa = (bits >> 16) & 0x7F;
    if (exp <= 0) out[i] = (uint8_t)sign;
    else if (exp >= 15) out[i] = (uint8_t)(sign | 0x7F);
    else out[i] = (uint8_t)(sign | (exp << 3) | (mantissa >> 4));
}

/* ========================================================================
 * Cache struct (opaque, only here we know the internals)
 * ======================================================================== */

struct ds4_kv_cache {
    uint32_t n_ctx, n_layers, n_kv_heads, head_dim;
    uint8_t *d_k_cache;
    uint8_t *d_v_cache;
    uint32_t cur_len;
    uint64_t total_bytes;
};

/* ========================================================================
 * GPU Management Functions
 * ======================================================================== */

extern "C" ds4_kv_cache *ds4_kv_cache_alloc_gpu(ds4_kv_cache *cache) {
    if (!cache) return NULL;

    uint64_t row = (uint64_t)cache->n_kv_heads * cache->head_dim;
    uint64_t layer = (uint64_t)cache->n_ctx * row;

    cudaError_t e1 = cudaMalloc(&cache->d_k_cache, layer * cache->n_layers);
    cudaError_t e2 = cudaMalloc(&cache->d_v_cache, layer * cache->n_layers);
    if (e1 != cudaSuccess || e2 != cudaSuccess) {
        fprintf(stderr, "KV Cache: GPU alloc failed (%s)\n",
                cudaGetErrorString(e1 != cudaSuccess ? e1 : e2));
        return NULL;
    }

    cudaMemset(cache->d_k_cache, 0, layer * cache->n_layers);
    cudaMemset(cache->d_v_cache, 0, layer * cache->n_layers);

    fprintf(stderr, "  GPU memory allocated: %.2f MiB\n",
            cache->total_bytes / (1024.0 * 1024.0));
    return cache;
}

extern "C" void ds4_kv_cache_free_gpu(ds4_kv_cache *cache) {
    if (!cache) return;
    if (cache->d_k_cache) cudaFree(cache->d_k_cache);
    if (cache->d_v_cache) cudaFree(cache->d_v_cache);
}

extern "C" void ds4_kv_cache_reset_gpu(ds4_kv_cache *cache) {
    if (!cache) return;
    uint64_t row = (uint64_t)cache->n_kv_heads * cache->head_dim;
    uint64_t layer = (uint64_t)cache->n_ctx * row;
    cudaMemset(cache->d_k_cache, 0, layer * cache->n_layers);
    cudaMemset(cache->d_v_cache, 0, layer * cache->n_layers);
}

extern "C" void ds4_kv_cache_append_gpu(
    ds4_kv_cache *cache, uint32_t layer, uint32_t pos,
    const float *d_k, const float *d_v)
{
    if (!cache || layer >= cache->n_layers || pos >= cache->n_ctx) return;

    uint64_t row = (uint64_t)cache->n_kv_heads * cache->head_dim;
    uint64_t layer_off = (uint64_t)layer * cache->n_ctx * row;
    uint64_t row_off = (uint64_t)pos * row;

    int threads = 256;
    int blocks = (int)((row + threads - 1) / threads);

    fp32_to_fp8_kernel<<<blocks, threads>>>(
        d_k, cache->d_k_cache + layer_off + row_off, (int)row);
    fp32_to_fp8_kernel<<<blocks, threads>>>(
        d_v, cache->d_v_cache + layer_off + row_off, (int)row);
}

extern "C" void ds4_kv_cache_get_ptrs_gpu(
    ds4_kv_cache *cache, uint32_t layer,
    void **d_k, void **d_v, uint32_t *stride)
{
    if (!cache || layer >= cache->n_layers) {
        if (d_k) *d_k = NULL;
        if (d_v) *d_v = NULL;
        if (stride) *stride = 0;
        return;
    }

    uint64_t row = (uint64_t)cache->n_kv_heads * cache->head_dim;
    uint64_t layer_off = (uint64_t)layer * cache->n_ctx * row;

    if (d_k) *d_k = cache->d_k_cache + layer_off;
    if (d_v) *d_v = cache->d_v_cache + layer_off;
    if (stride) *stride = (uint32_t)row;
}
