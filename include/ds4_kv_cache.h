#ifndef DS4_KV_CACHE_H
#define DS4_KV_CACHE_H

#ifdef __CUDACC__
#define CUDA_HOST_DEVICE __host__ __device__
#else
#define CUDA_HOST_DEVICE
#endif

#ifdef __cplusplus
extern "C" {
#endif

#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

typedef struct ds4_kv_cache ds4_kv_cache;

static inline CUDA_HOST_DEVICE float ds4_f8e4m3_to_f32(uint8_t v) {
    if (!v) return 0.0f;
    unsigned s = (v >> 7) & 1;
    unsigned e = (v >> 3) & 0xF;
    unsigned m = v & 0x7;
    float val = (e == 0) ? (float)m / 64.0f
                         : (float)(m + 8) / 8.0f * ldexpf(1.0f, (int)e - 7);
    return s ? -val : val;
}

static inline uint8_t ds4_f32_to_f8e4m3(float f) {
    if (f == 0.0f) return 0;
    unsigned sign = f < 0 ? 1 : 0;
    f = fabsf(f);
    int mantissa, exp;
    mantissa = (int)(f * 8.0f);
    exp = 0;
    while (mantissa >= 16) { mantissa >>= 1; exp++; }
    while (mantissa < 8 && exp > 0) { mantissa <<= 1; exp--; }
    if (exp > 15) return sign ? 0x80 | 0xF : 0xF;
    if (exp < 0) return 0;
    return (uint8_t)(sign << 7) | ((uint8_t)exp << 3) | (uint8_t)(mantissa & 7);
}

ds4_kv_cache *ds4_kv_cache_create(uint32_t max_seq_len, uint32_t n_layers,
                                 uint32_t n_kv_heads, uint32_t head_dim);
void ds4_kv_cache_free(ds4_kv_cache *cache);
void ds4_kv_cache_reset(ds4_kv_cache *cache);
void ds4_kv_cache_set_fp8(
    ds4_kv_cache *cache, uint32_t layer, uint32_t pos,
    const uint8_t *d_k, const uint8_t *d_v);
void ds4_kv_cache_append(
    ds4_kv_cache *cache, uint32_t layer, uint32_t pos,
    const float *d_k, const float *d_v);
void ds4_kv_cache_get_fp8_ptrs(
    ds4_kv_cache *cache, uint32_t layer,
    void **d_k_ptr, void **d_v_ptr, uint32_t *row_stride);
uint32_t ds4_kv_cache_len(const ds4_kv_cache *cache);
uint64_t ds4_kv_cache_memory_bytes(const ds4_kv_cache *cache);
void ds4_kv_cache_stats(const ds4_kv_cache *cache, FILE *fp);

#ifdef __cplusplus
}
#endif

#endif /* DS4_KV_CACHE_H */
