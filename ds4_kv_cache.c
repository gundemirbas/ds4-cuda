/**
 * ds4_kv_cache.c — FP8 KV Cache implementation (CPU-side management)
 */

#include "ds4_kv_cache.h"
#include <stdlib.h>
#include <string.h>

/* The actual cache struct is defined in the .cu file for CUDA access */
struct ds4_kv_cache {
    uint32_t n_ctx;
    uint32_t n_layers;
    uint32_t n_kv_heads;
    uint32_t head_dim;
    void *d_k_cache;   /* GPU pointer */
    void *d_v_cache;   /* GPU pointer */
    uint32_t cur_len;
    uint64_t total_bytes;
};

/* CUDA functions (defined in ds4_kv_cache.cu) */
extern ds4_kv_cache *ds4_kv_cache_alloc_gpu(ds4_kv_cache *cache);
extern void ds4_kv_cache_free_gpu(ds4_kv_cache *cache);
extern void ds4_kv_cache_reset_gpu(ds4_kv_cache *cache);
extern void ds4_kv_cache_append_gpu(ds4_kv_cache *cache, uint32_t layer,
                                     uint32_t pos, const float *d_k, const float *d_v);
extern void ds4_kv_cache_get_ptrs_gpu(ds4_kv_cache *cache, uint32_t layer,
                                       void **d_k, void **d_v, uint32_t *stride);

ds4_kv_cache *ds4_kv_cache_create(
    uint32_t n_ctx, uint32_t n_layers,
    uint32_t n_kv_heads, uint32_t head_dim)
{
    ds4_kv_cache *cache = (ds4_kv_cache *)calloc(1, sizeof(ds4_kv_cache));
    if (!cache) return NULL;

    cache->n_ctx = n_ctx;
    cache->n_layers = n_layers;
    cache->n_kv_heads = n_kv_heads;
    cache->head_dim = head_dim;

    /* FP8: 1 byte per element */
    uint64_t row = (uint64_t)n_kv_heads * head_dim;
    uint64_t layer = (uint64_t)n_ctx * row;
    cache->total_bytes = 2 * (uint64_t)n_layers * layer;

    fprintf(stderr, "KV Cache (FP8 E4M3): %u layers × %u ctx × %u heads × %u dim\n",
            n_layers, n_ctx, n_kv_heads, head_dim);
    fprintf(stderr, "  Total: %.2f MiB (vs FP32: %.2f MiB, 4× savings)\n",
            cache->total_bytes / (1024.0 * 1024.0),
            cache->total_bytes * 4 / (1024.0 * 1024.0));

    return ds4_kv_cache_alloc_gpu(cache);
}

void ds4_kv_cache_free(ds4_kv_cache *cache) {
    if (!cache) return;
    ds4_kv_cache_free_gpu(cache);
    free(cache);
}

void ds4_kv_cache_reset(ds4_kv_cache *cache) {
    if (!cache) return;
    ds4_kv_cache_reset_gpu(cache);
    cache->cur_len = 0;
}

void ds4_kv_cache_append(
    ds4_kv_cache *cache, uint32_t layer, uint32_t pos,
    const float *d_k, const float *d_v)
{
    if (!cache) return;
    ds4_kv_cache_append_gpu(cache, layer, pos, d_k, d_v);
    if (pos + 1 > cache->cur_len) cache->cur_len = pos + 1;
}

void ds4_kv_cache_get_fp8_ptrs(
    ds4_kv_cache *cache, uint32_t layer,
    void **d_k_ptr, void **d_v_ptr, uint32_t *row_stride)
{
    if (!cache) {
        if (d_k_ptr) *d_k_ptr = NULL;
        if (d_v_ptr) *d_v_ptr = NULL;
        if (row_stride) *row_stride = 0;
        return;
    }
    ds4_kv_cache_get_ptrs_gpu(cache, layer, d_k_ptr, d_v_ptr, row_stride);
}

uint32_t ds4_kv_cache_len(const ds4_kv_cache *cache) {
    return cache ? cache->cur_len : 0;
}

uint64_t ds4_kv_cache_memory_bytes(const ds4_kv_cache *cache) {
    return cache ? cache->total_bytes : 0;
}

void ds4_kv_cache_stats(const ds4_kv_cache *cache, FILE *fp) {
    if (!cache) { fprintf(fp, "KV Cache: NULL\n"); return; }
    fprintf(fp, "KV Cache (FP8): %u layers × %u ctx × %u heads × %u dim, len=%u/%u, %.2f MiB\n",
            cache->n_layers, cache->n_ctx, cache->n_kv_heads, cache->head_dim,
            cache->cur_len, cache->n_ctx,
            cache->total_bytes / (1024.0 * 1024.0));
}
