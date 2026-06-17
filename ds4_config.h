/* ds4_config.h — config.json parser for ds4-cuda
 *
 * ds4_variant ve ds4_shape tanımları src/ds4.c'de yapıyoruz.
 * Bu header sadece config.json parsed result'ını tutar. */
#ifndef DS4_CONFIG_H
#define DS4_CONFIG_H

#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Parsed config.json — weight loading ve model identification için */
typedef struct {
    const char *name;
    uint32_t n_layer;
    uint32_t n_embd;
    uint32_t n_vocab;
    uint32_t n_head;
    uint32_t n_kv_heads;
    uint32_t head_dim;
    uint32_t n_expert;
    uint32_t n_expert_used;
    uint32_t n_expert_shared;
    uint32_t n_expert_groups;
    uint32_t n_group_used;
    uint32_t n_ff_exp;
    uint32_t n_hash_layer;
    uint32_t n_swa;
    uint32_t n_indexer_heads;
    uint32_t n_indexer_head_dim;
    uint32_t n_indexer_top_k;
    uint32_t n_hc;
    uint32_t n_hc_sinkhorn_iter;
    uint32_t n_lora_q;
    uint32_t n_lora_o;
    uint64_t max_seq_len;
    float rms_eps;
    float hc_eps;
    float expert_weight_scale;
    float swiglu_clamp_exp;
    float rope_freq_base;
    float rope_scale_factor;
    float rope_yarn_beta_fast;
    float rope_yarn_beta_slow;
    float compress_rope_freq_base;
    uint64_t rope_orig_ctx;
} ds4_config;

/* Load config.json from model_dir, parse it, return filled ds4_config.
 * Returns 1 on success, 0 on failure. */
int ds4_config_load(const char *model_dir, ds4_config *cfg);

#ifdef __cplusplus
}
#endif

#endif /* DS4_CONFIG_H */
