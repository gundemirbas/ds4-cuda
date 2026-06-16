#ifndef DS4_SSD_STREAMING_H
#define DS4_SSD_STREAMING_H
#include <stdbool.h>
#include <stdint.h>

typedef struct {
    uint32_t layer_idx;
    uint32_t expert_idx;
    uint64_t last_access;
    uint32_t access_count;
    bool is_hot;
    void *weight_gate;
    void *weight_up;
    void *weight_down;
    uint64_t weight_size;
    bool is_loaded;
} ssd_expert_entry;

typedef struct {
    ssd_expert_entry *entries;
    uint32_t capacity;
    uint32_t count;
    uint64_t total_bytes;
    uint64_t max_bytes;
    uint64_t hit_count;
    uint64_t miss_count;
} ssd_expert_cache;

ssd_expert_cache *ssd_expert_cache_create(uint32_t capacity, uint64_t max_bytes);
void ssd_expert_cache_free(ssd_expert_cache *cache);
ssd_expert_entry *ssd_expert_cache_get(ssd_expert_cache *cache, uint32_t layer, uint32_t expert);
int ssd_expert_cache_put(ssd_expert_cache *cache, uint32_t layer, uint32_t expert, void *data, uint64_t size);
void ssd_expert_cache_stats(ssd_expert_cache *cache, uint64_t *hits, uint64_t *misses, double *hit_rate);

#endif
