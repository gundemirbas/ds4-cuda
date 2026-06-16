/**
 * ds4 Expert Cache
 *
 * LRU cache for MoE experts with hot/cold management.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "ds4_ssd_streaming.h"

/* ========================================================================
 * Expert Cache Implementation
 * ======================================================================== */

ssd_expert_cache *ssd_expert_cache_create(uint32_t capacity, uint64_t max_bytes) {
    ssd_expert_cache *cache = calloc(1, sizeof(ssd_expert_cache));
    if (!cache) return NULL;

    cache->entries = calloc(capacity, sizeof(ssd_expert_entry));
    if (!cache->entries) {
        free(cache);
        return NULL;
    }

    cache->capacity = capacity;
    cache->count = 0;
    cache->total_bytes = 0;
    cache->max_bytes = max_bytes;
    cache->hit_count = 0;
    cache->miss_count = 0;

    for (uint32_t i = 0; i < capacity; i++) {
        cache->entries[i].is_loaded = false;
        cache->entries[i].is_hot = false;
    }

    return cache;
}

void ssd_expert_cache_free(ssd_expert_cache *cache) {
    if (!cache) return;

    for (uint32_t i = 0; i < cache->capacity; i++) {
        if (cache->entries[i].is_loaded) {
            free(cache->entries[i].weight_gate);
            free(cache->entries[i].weight_up);
            free(cache->entries[i].weight_down);
        }
    }

    free(cache->entries);
    free(cache);
}

static ssd_expert_entry *find_entry(ssd_expert_cache *cache, uint32_t layer, uint32_t expert) {
    for (uint32_t i = 0; i < cache->capacity; i++) {
        ssd_expert_entry *e = &cache->entries[i];
        if (e->is_loaded && e->layer_idx == layer && e->expert_idx == expert) {
            return e;
        }
    }
    return NULL;
}

static ssd_expert_entry *find_empty(ssd_expert_cache *cache) {
    for (uint32_t i = 0; i < cache->capacity; i++) {
        if (!cache->entries[i].is_loaded) {
            return &cache->entries[i];
        }
    }
    return NULL;
}

static ssd_expert_entry *find_victim(ssd_expert_cache *cache) {
    ssd_expert_entry *victim = NULL;
    uint64_t oldest_time = UINT64_MAX;

    for (uint32_t i = 0; i < cache->capacity; i++) {
        ssd_expert_entry *e = &cache->entries[i];
        if (!e->is_loaded) continue;
        if (e->is_hot) continue;

        if (e->last_access < oldest_time) {
            oldest_time = e->last_access;
            victim = e;
        }
    }

    return victim;
}

static void evict_entry(ssd_expert_cache *cache, ssd_expert_entry *entry) {
    if (!entry || !entry->is_loaded) return;

    free(entry->weight_gate);
    free(entry->weight_up);
    free(entry->weight_down);

    cache->total_bytes -= entry->weight_size;
    cache->count--;

    entry->is_loaded = false;
    entry->is_hot = false;
    entry->weight_gate = NULL;
    entry->weight_up = NULL;
    entry->weight_down = NULL;
    entry->weight_size = 0;
}

ssd_expert_entry *ssd_expert_cache_get(ssd_expert_cache *cache, uint32_t layer, uint32_t expert) {
    if (!cache) return NULL;

    ssd_expert_entry *entry = find_entry(cache, layer, expert);

    if (entry) {
        entry->last_access = ++cache->hit_count;
        entry->access_count++;

        if (entry->access_count >= 3 && !entry->is_hot) {
            entry->is_hot = true;
        }

        return entry;
    }

    cache->miss_count++;
    return NULL;
}

int ssd_expert_cache_put(ssd_expert_cache *cache, uint32_t layer, uint32_t expert,
                         void *data, uint64_t size) {
    if (!cache || !data) return -1;

    ssd_expert_entry *existing = find_entry(cache, layer, expert);
    if (existing) {
        existing->last_access = ++cache->hit_count;
        existing->access_count++;
        return 0;
    }

    ssd_expert_entry *entry = find_empty(cache);

    if (!entry) {
        entry = find_victim(cache);
        if (!entry) return -1;
        evict_entry(cache, entry);
    }

    entry->layer_idx = layer;
    entry->expert_idx = expert;
    entry->last_access = ++cache->hit_count;
    entry->access_count = 1;
    entry->is_hot = false;
    entry->is_loaded = true;
    entry->weight_size = size;

    entry->weight_gate = malloc(size);
    if (!entry->weight_gate) {
        entry->is_loaded = false;
        return -1;
    }
    memcpy(entry->weight_gate, data, size);
    entry->weight_up = NULL;
    entry->weight_down = NULL;

    cache->count++;
    cache->total_bytes += size;

    return 0;
}

void ssd_expert_cache_stats(ssd_expert_cache *cache, uint64_t *hits, uint64_t *misses, double *hit_rate) {
    if (!cache) {
        if (hits) *hits = 0;
        if (misses) *misses = 0;
        if (hit_rate) *hit_rate = 0.0;
        return;
    }

    if (hits) *hits = cache->hit_count;
    if (misses) *misses = cache->miss_count;
    if (hit_rate) {
        uint64_t total = cache->hit_count + cache->miss_count;
        *hit_rate = (total > 0) ? (double)cache->hit_count / total : 0.0;
    }
}
