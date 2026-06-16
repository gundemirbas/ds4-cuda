/**
 * ds4_main.c — Full inference pipeline with NVFP4 GPU forward pass
 *
 * Integrates:
 *  - Safetensors model loading
 *  - FP8 KV cache for all 43 layers
 *  - Token embedding (FP8 lookup)
 *  - 43-layer transformer forward pass (GPU)
 *  - Output projection (FP8 GEMV)
 *  - Token sampling (argmax)
 */

#define _GNU_SOURCE
#define _POSIX_C_SOURCE 199309L

#include <errno.h>
#include <fcntl.h>
#include <float.h>
#include <inttypes.h>
#include <math.h>
#include <pthread.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>

#include "ds4_config.h"
#include "ds4_safetensors.h"
#include "ds4_kv_cache.h"

ds4_kv_cache *ds4_kv_cache_create(uint32_t max_seq_len, uint32_t n_layers,
                                 uint32_t n_kv_heads, uint32_t head_dim);
void ds4_kv_cache_free(ds4_kv_cache *cache);
void ds4_kv_cache_reset(ds4_kv_cache *cache);
uint64_t ds4_kv_cache_memory_bytes(const ds4_kv_cache *cache);



// External function declarations (implemented in other compilation units)
extern "C" void ds4_layer_forward(
    const void *lw, ds4_kv_cache *kv_cache, uint32_t layer_idx,
    float *d_in, float *d_out, int pos,
    float *scratch, float eps, float ws2, float freq_base);

extern "C" void launch_token_embedding(const uint8_t *d_embedding, const int32_t *d_tokens,
                            float *d_output, int n_vocab, int n_embd, int n_tokens);
extern "C" void launch_output_projection(const float *d_hidden, const uint8_t *d_output_w,
                              float *d_logits, int n_vocab, int n_embd);
extern "C" void launch_fp8_attention(const float *d_q, const uint8_t *d_k_cache,
                          const uint8_t *d_v_cache, float *d_scores,
                          float *d_output, int n_heads, int head_dim,
                          int seq_len, int n_kv_heads, int pos);



/* CUDA kernel declarations */


/* ========================================================================
 * Logging
 * ======================================================================== */

bool ds4_log_is_tty(FILE *fp) { return isatty(fileno(fp)); }

void ds4_log(FILE *fp, ds4_log_type type, const char *fmt, ...) {
    const char *tag = "";
    switch (type) {
    case DS4_LOG_OK:          tag = "[OK]"; break;
    case DS4_LOG_ERROR:       tag = "[ERR]"; break;
    case DS4_LOG_WARNING:     tag = "[WARN]"; break;
    case DS4_LOG_TIMING:      tag = "[TIME]"; break;
    case DS4_LOG_PREFILL:     tag = "[PF]"; break;
    case DS4_LOG_GENERATION:  tag = "[GEN]"; break;
    case DS4_LOG_KVCACHE:     tag = "[KV]"; break;
    case DS4_LOG_TOOL:        tag = "[TOOL]"; break;
    default:                  tag = "[...]"; break;
    }
    va_list ap;
    va_start(ap, fmt);
    fprintf(fp, "%s ", tag);
    vfprintf(fp, fmt, ap);
    fprintf(fp, "\n");
    va_end(ap);
}


static float bf16_val(uint16_t v) {
    unsigned s=(v>>15)&1, e=(v>>7)&0xFF, m=v&0x7F;
    float val;
    if (e==0) val = ldexpf((float)m, -126);
    else if (e==0xFF) val = (m==0) ? INFINITY : NAN;
    else val = ldexpf(1.f + (float)m/128.f, (int)e-127);
    return s ? -val : val;
}

static void ds4_die(const char *msg) {
    fprintf(stderr, "ds4: %s\n", msg);
    exit(1);
}

static void *xmalloc(size_t n) {
    void *p = malloc(n);
    if (!p && n) ds4_die("out of memory");
    return p;
}

static void *xcalloc(size_t count, size_t sz) {
    void *p = calloc(count, sz);
    if (!p && count && sz) ds4_die("out of memory");
    return p;
}

static char *xstrdup(const char *s) {
    size_t len = strlen(s);
    char *p = (char *)malloc(len + 1);
    if (!p) ds4_die("out of memory");
    memcpy(p, s, len + 1);
    return p;
}

/* ========================================================================
 * Timer
 * ======================================================================== */

static double now_sec(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

/* ========================================================================
 * Token helpers
 * ======================================================================== */

void ds4_tokens_push(ds4_tokens *tv, int token) {
    if (tv->len >= tv->cap) {
        tv->cap = tv->cap ? tv->cap * 2 : 256;
        tv->v = (int *)realloc(tv->v, (size_t)tv->cap * sizeof(int));
    }
    tv->v[tv->len++] = token;
}

void ds4_tokens_free(ds4_tokens *tv) {
    free(tv->v);
    tv->v = NULL;
    tv->len = tv->cap = 0;
}

void ds4_tokens_copy(ds4_tokens *dst, const ds4_tokens *src) {
    ds4_tokens_free(dst);
    dst->cap = src->len;
    dst->len = src->len;
    dst->v = (int *)xmalloc((size_t)src->len * sizeof(int));
    memcpy(dst->v, src->v, (size_t)src->len * sizeof(int));
}

bool ds4_tokens_starts_with(const ds4_tokens *tokens, const ds4_tokens *prefix) {
    if (tokens->len < prefix->len) return false;
    return memcmp(tokens->v, prefix->v, (size_t)prefix->len * sizeof(int)) == 0;
}

/* ========================================================================
 * Backend info
 * ======================================================================== */

const char *ds4_backend_name(ds4_backend backend) {
    switch (backend) {
    case DS4_BACKEND_CUDA:  return "CUDA";
    case DS4_BACKEND_METAL: return "Metal";
    case DS4_BACKEND_CPU:   return "CPU";
    }
    return "unknown";
}

/* ========================================================================
 * Think mode
 * ======================================================================== */

bool ds4_think_mode_enabled(ds4_think_mode mode) { return mode != DS4_THINK_NONE; }

const char *ds4_think_mode_name(ds4_think_mode mode) {
    switch (mode) {
    case DS4_THINK_NONE:  return "none";
    case DS4_THINK_HIGH:  return "high";
    case DS4_THINK_MAX:   return "max";
    }
    return "unknown";
}

const char *ds4_think_max_prefix(void) { return " thinking"; }
uint32_t ds4_think_max_min_context(void) { return 4096; }

ds4_think_mode ds4_think_mode_for_context(ds4_think_mode mode, int ctx_size) {
    if (mode == DS4_THINK_MAX && ctx_size < (int)ds4_think_max_min_context())
        return DS4_THINK_HIGH;
    return mode;
}

/* ========================================================================
 * SSD helpers (from ds4_ssd.h)
 * ======================================================================== */

bool ds4_parse_gib_arg(const char *s, uint64_t *bytes) {
    char *end;
    double v = strtod(s, &end);
    if (end == s) return false;
    if (*end == 'G' || *end == 'g') end++;
    if (*end == 'i' && (end[1] == 'B' || end[1] == 'b')) end += 2;
    *bytes = (uint64_t)(v * 1024.0 * 1024.0 * 1024.0);
    return true;
}

bool ds4_parse_streaming_cache_experts_arg(const char *s, uint32_t *experts, uint64_t *bytes) {
    *experts = (uint32_t)atoi(s);
    *bytes = 0;
    return *experts > 0;
}

uint32_t ds4_ssd_cache_experts_for_byte_budget(uint64_t bytes, uint64_t per_expert_bytes) {
    if (per_expert_bytes == 0) return 0;
    return (uint32_t)(bytes / per_expert_bytes);
}

bool ds4_ssd_auto_cache_plan(uint64_t recommended_bytes, uint64_t non_routed_bytes,
                              uint64_t per_expert_bytes, uint64_t max_model_experts,
                              ds4_ssd_cache_plan *out) {
    uint64_t cache_budget = recommended_bytes > non_routed_bytes
        ? recommended_bytes - non_routed_bytes : 0;
    uint32_t experts = ds4_ssd_cache_experts_for_byte_budget(cache_budget, per_expert_bytes);
    if (experts > max_model_experts) experts = (uint32_t)max_model_experts;
    out->model_target_bytes = recommended_bytes;
    out->cache_bytes = (uint64_t)experts * per_expert_bytes;
    out->effective_cache_bytes = out->cache_bytes;
    out->cache_experts = experts;
    return true;
}

bool ds4_ssd_memory_lock_acquire(ds4_ssd_memory_lock *lock, uint64_t bytes) {
    lock->ptr = malloc(bytes);
    lock->bytes = lock->ptr ? bytes : 0;
    return lock->ptr != NULL;
}

void ds4_ssd_memory_lock_release(ds4_ssd_memory_lock *lock) {
    free(lock->ptr);
    lock->ptr = NULL;
    lock->bytes = 0;
}

/* ========================================================================
 * Context memory estimate
 * ======================================================================== */

ds4_context_memory ds4_context_memory_estimate(ds4_backend backend, int ctx_size) {
    return ds4_context_memory_estimate_with_prefill(backend, ctx_size, 4096);
}

ds4_context_memory ds4_context_memory_estimate_with_prefill(
        ds4_backend backend, int ctx_size, uint32_t prefill_chunk) {
    (void)backend;
    uint64_t kv_bytes = (uint64_t)ctx_size * 2 * 1 * 512; /* FP8 KV cache */
    uint64_t scratch = 256 * 1024 * 1024;
    ds4_context_memory m = {
        .total_bytes = kv_bytes + scratch,
        .raw_bytes = kv_bytes,
        .compressed_bytes = 0,
        .scratch_bytes = scratch,
        .prefill_cap = prefill_chunk,
        .raw_cap = (uint32_t)(kv_bytes / 1024), /* bytes per token */
        .comp_cap = 0,
    };
    return m;
}

/* ========================================================================
 * Engine struct
 * ======================================================================== */

struct ds4_engine {
    char *model_path;
    ds4_engine_options options;
    ds4_shape shape;
    bool loaded;
    sst_sharded_model *sst;
    int32_t *vocab_ids;
    char   **vocab_text;
    int n_vocab;
    int eos_id;
    int user_id;
    int assistant_id;
    /* GPU device pointers for all layers */
    uint8_t *d_embedding;   /* embedding table [n_vocab][n_embd] FP8 */
    uint8_t *d_output_w;    /* output weights [n_vocab][n_embd] FP8 */
    /* Per-layer weight pointers (loaded once, reused across sessions) */
    void *d_layer_weights;  /* array of ds4_layer_weights structs */
};

struct ds4_session {
    ds4_engine *engine;
    ds4_tokens tokens;
    int ctx_size;
    int pos;
    float *logits;
    ds4_kv_cache *kv_cache;
    float *d_scratch;       /* GPU scratch buffer */
    int32_t *d_tokens;      /* GPU token buffer */
    ds4_session_progress_fn progress_fn;
    void *progress_ud;
    ds4_session_cancel_fn cancel_fn;
    void *cancel_ud;
    ds4_ssd_memory_lock simulated_memory;
};

/* ========================================================================
 * Layer weights loader
 * ======================================================================== */

typedef struct {
    uint8_t *wq_a, *wq_b, *wkv, *wo_a, *wo_b;
    float *attn_norm, *ffn_norm;
    float *gate;
    uint8_t *w1, *w1s, *w2, *w2s, *w3, *w3s;
} layer_weights_t;

static void load_layer_weights(sst_sharded_model *sst, uint32_t layer_idx, layer_weights_t *lw) {
    char name[128];
    // Helper macro: load F8 weight, allocate GPU memory, copy
    #define LOAD_F8_WEIGHT(dst, tensor_name, size) do { \
        snprintf(name, sizeof(name), tensor_name, layer_idx); \
        sst_tensor *t = sst_sharded_model_find_tensor(sst, name); \
        if (!t) { fprintf(stderr, "[WARN] Tensor %s not found\n", name); dst = NULL; break; } \
        void *src = sst_sharded_model_tensor_data(sst, t); \
        cudaMalloc(&dst, size); \
        cudaMemcpy(dst, src, size, cudaMemcpyHostToDevice); \
        fprintf(stderr, "[DEBUG] Loaded %s -> GPU %p\n", name, dst); fflush(stderr); \
    } while(0)

    LOAD_F8_WEIGHT(lw->wq_a, "layers.%u.attn.wq_a.weight", 1024 * 4096);
    LOAD_F8_WEIGHT(lw->wq_b, "layers.%u.attn.wq_b.weight", 32768 * 1024);
    LOAD_F8_WEIGHT(lw->wkv,  "layers.%u.attn.wkv.weight",  512 * 4096);
    LOAD_F8_WEIGHT(lw->wo_a, "layers.%u.attn.wo_a.weight",  8192 * 4096);
    LOAD_F8_WEIGHT(lw->wo_b, "layers.%u.attn.wo_b.weight",  4096 * 8192);

    // Gate weights: BF16 -> F32 conversion, allocate GPU memory
    snprintf(name, sizeof(name), "layers.%u.ffn.gate.weight", layer_idx);
    uint16_t *gate_u16 = (uint16_t *)sst_sharded_model_tensor_data(sst, sst_sharded_model_find_tensor(sst, name));
    float *gate_f32 = (float *)malloc(256 * 4096 * sizeof(float));
    for (int i = 0; i < 256 * 4096; i++) gate_f32[i] = bf16_val(gate_u16[i]);
    cudaMalloc(&lw->gate, 256 * 4096 * sizeof(float));
    cudaMemcpy(lw->gate, gate_f32, 256 * 4096 * sizeof(float), cudaMemcpyHostToDevice);
    free(gate_f32);
    fprintf(stderr, "[DEBUG] Loaded gate -> GPU %p\n", lw->gate); fflush(stderr);

    // Norm weights: BF16 -> F32 conversion, allocate GPU memory
    #define LOAD_NORM_WEIGHT(dst, tensor_name) do { \
        snprintf(name, sizeof(name), tensor_name, layer_idx); \
        uint16_t *src_u16 = (uint16_t *)sst_sharded_model_tensor_data(sst, sst_sharded_model_find_tensor(sst, name)); \
        float *tmp_f32 = (float *)malloc(4096 * sizeof(float)); \
        for (int i = 0; i < 4096; i++) tmp_f32[i] = bf16_val(src_u16[i]); \
        cudaMalloc(&dst, 4096 * sizeof(float)); \
        cudaMemcpy(dst, tmp_f32, 4096 * sizeof(float), cudaMemcpyHostToDevice); \
        free(tmp_f32); \
        fprintf(stderr, "[DEBUG] Loaded %s -> GPU %p\n", name, dst); fflush(stderr); \
    } while(0)

    LOAD_NORM_WEIGHT(lw->attn_norm, "layers.%u.attn_norm.weight");
    LOAD_NORM_WEIGHT(lw->ffn_norm,  "layers.%u.ffn_norm.weight");

    LOAD_F8_WEIGHT(lw->w1,  "layers.%u.ffn.experts.0.w1.weight",           4096 * 1024);
    LOAD_F8_WEIGHT(lw->w1s, "layers.%u.ffn.experts.0.w1.weight_scale",     4096 * 128);
    LOAD_F8_WEIGHT(lw->w2,  "layers.%u.ffn.experts.0.w2.weight",           4096 * 1024);
    LOAD_F8_WEIGHT(lw->w2s, "layers.%u.ffn.experts.0.w2.weight_scale",     4096 * 128);
    LOAD_F8_WEIGHT(lw->w3,  "layers.%u.ffn.experts.0.w3.weight",           4096 * 1024);
    LOAD_F8_WEIGHT(lw->w3s, "layers.%u.ffn.experts.0.w3.weight_scale",     4096 * 128);

    #undef LOAD_F8_WEIGHT
    #undef LOAD_NORM_WEIGHT
}

/* ========================================================================
 * Engine open / close
 * ======================================================================== */

static const ds4_shape DS4_SHAPE_FLASH = {
    .name = "DeepSeek V4 Flash",
    .variant = DS4_VARIANT_FLASH,
    .n_layer = 43,
    .n_embd = 4096,
    .n_vocab = 129280,
    .n_head = 64,
    .n_head_kv = 1,
    .n_head_dim = 512,
    .n_value_dim = 512,
    .n_rot = 64,
    .n_out_group = 8,
    .n_lora_q = 1024,
    .n_lora_o = 1024,
    .n_expert = 256,
    .n_expert_used = 6,
    .n_expert_shared = 1,
    .n_ff_exp = 3072,
    .n_hash_layer = 3,
    .n_swa = 128,
    .n_indexer_head = 64,
    .n_indexer_head_dim = 128,
    .n_indexer_top_k = 1024,
    .n_hc = 4,
    .n_hc_sinkhorn_iter = 20,
    .rms_eps = 1.0e-5f,
    .hc_eps = 1.0e-5f,
    .expert_weight_scale = 2.0f,
    .swiglu_clamp_exp = 7.0f,
    .rope_freq_base = 10000.0f,
    .rope_scale_factor = 1.0f,
    .rope_yarn_beta_fast = 1.0f,
    .rope_yarn_beta_slow = 1.0f,
    .compress_rope_freq_base = 10000.0f,
    .rope_orig_ctx = 0,
};

static void preload_all_layers(ds4_engine *e);

int ds4_engine_open(ds4_engine **out, const ds4_engine_options *opt) {
    ds4_engine *e = (ds4_engine *)xcalloc(1, sizeof(ds4_engine));
    e->model_path = xstrdup(opt->model_path);
    e->options = *opt;
    e->shape = DS4_SHAPE_FLASH;
    e->loaded = false;

    fprintf(stderr, "[ds4] Loading safetensors from %s...\n", opt->model_path);
    e->sst = sst_sharded_model_load(opt->model_path);
    if (!e->sst) {
        fprintf(stderr, "[ds4] Failed to load model from %s\n", opt->model_path);
        free(e->model_path);
        free(e);
        return 1;
    }
    e->loaded = true;
    fprintf(stderr, "[ds4] Model loaded: %s (NVFP4)\n", e->shape.name);

    /* Load embedding table */
    sst_tensor *t = sst_sharded_model_find_tensor(e->sst, "embed.weight");
    if (t) {
        uint8_t *embedding = (uint8_t *)sst_sharded_model_tensor_data(e->sst, t);
        size_t emb_bytes = (size_t)e->shape.n_vocab * e->shape.n_embd;
        if (cudaMalloc(&e->d_embedding, emb_bytes) != cudaSuccess) { fprintf(stderr, "Failed to allocate embedding table\n"); return 1; }
        cudaMemcpy(e->d_embedding, embedding, emb_bytes, cudaMemcpyHostToDevice);
        fprintf(stderr, "[ds4] Embedding table: %u x %u (FP8) = %.2f MiB\n",
                e->shape.n_vocab, e->shape.n_embd, emb_bytes / (1024.0 * 1024.0));
    } else {
        fprintf(stderr, "[ds4] WARNING: no embedding table found\n");
    }

    /* Load output weights */
    t = sst_sharded_model_find_tensor(e->sst, "head.weight");
    if (t) {
        uint8_t *output_w = (uint8_t *)sst_sharded_model_tensor_data(e->sst, t);
        size_t out_bytes = (size_t)e->shape.n_vocab * e->shape.n_embd;
        if (cudaMalloc(&e->d_output_w, out_bytes) != cudaSuccess) { fprintf(stderr, "Failed to allocate output weights\n"); return 1; }
        cudaMemcpy(e->d_output_w, output_w, out_bytes, cudaMemcpyHostToDevice);
        fprintf(stderr, "[ds4] Output weights: %u x %u (FP8) = %.2f MiB\n",
                e->shape.n_vocab, e->shape.n_embd, out_bytes / (1024.0 * 1024.0));
    } else {
        fprintf(stderr, "[ds4] WARNING: no output weights found\n");
    }

    /* Pre-load all layer weights to GPU */
    // preload_all_layers(e);  // TEST: skip preload

    *out = e;
    return 0;
}

void ds4_engine_close(ds4_engine *e) {
    if (!e) return;
    sst_sharded_model_free(e->sst);
    if (e->d_embedding) cudaFree(e->d_embedding);
    if (e->d_output_w) cudaFree(e->d_output_w);
    free(e->model_path);
    free(e);
}

void ds4_engine_summary(ds4_engine *e) {
    fprintf(stderr, "Model: %s\n", e->shape.name);
    fprintf(stderr, "  Layers: %u\n", e->shape.n_layer);
    fprintf(stderr, "  Hidden: %u\n", e->shape.n_embd);
    fprintf(stderr, "  Vocab:  %u\n", e->shape.n_vocab);
    fprintf(stderr, "  Heads:  %u (KV: %u)\n", e->shape.n_head, e->shape.n_head_kv);
    fprintf(stderr, "  Experts: %u (active: %u)\n", e->shape.n_expert, e->shape.n_expert_used);
    fprintf(stderr, "  Format: NVFP4 (E2M1 + E4M3 scales)\n");
}

int ds4_engine_vocab_size(ds4_engine *e) { return e->shape.n_vocab; }
int ds4_engine_power(ds4_engine *e) { (void)e; return 100; }
int ds4_engine_set_power(ds4_engine *e, int p) { (void)e; (void)p; return 0; }
const char *ds4_engine_model_name(ds4_engine *e) { return e->shape.name; }
int ds4_engine_layer_count(ds4_engine *e) { return e->shape.n_layer; }
uint32_t ds4_engine_layer_compress_ratio(ds4_engine *e, uint32_t l) { (void)e; (void)l; return 1; }
uint64_t ds4_engine_hidden_f32_values(ds4_engine *e) { return (uint64_t)e->shape.n_embd; }
int ds4_engine_model_id(ds4_engine *e) { (void)e; return 0; }
bool ds4_engine_has_output_head(ds4_engine *e) { (void)e; return false; }
bool ds4_engine_has_mtp(ds4_engine *e) { (void)e; return false; }
int ds4_engine_mtp_draft_tokens(ds4_engine *e) { (void)e; return 0; }
int ds4_engine_routed_quant_bits(ds4_engine *e) { (void)e; return 4; }

/* ========================================================================
 * Tokenizer stubs
 * ======================================================================== */

void ds4_tokenize_text(ds4_engine *e, const char *text, ds4_tokens *out) {
    (void)e;
    for (const char *p = text; *p; p++) {
        ds4_tokens_push(out, (int)(uint8_t)*p);
    }
}

void ds4_tokenize_rendered_chat(ds4_engine *e, const char *text, ds4_tokens *out) {
    ds4_tokenize_text(e, text, out);
}

void ds4_chat_begin(ds4_engine *e, ds4_tokens *tokens) {
    (void)e; tokens->len = 0;
}

void ds4_encode_chat_prompt(ds4_engine *e, const char *system, const char *prompt,
                            ds4_think_mode mode, ds4_tokens *out) {
    (void)mode;
    ds4_tokens_free(out);
    if (system && system[0]) ds4_tokenize_text(e, system, out);
    ds4_tokenize_text(e, prompt, out);
}

void ds4_chat_append_max_effort_prefix(ds4_engine *e, ds4_tokens *tokens) {
    (void)e; (void)tokens;
}

void ds4_chat_append_message(ds4_engine *e, ds4_tokens *tokens,
                             const char *role, const char *content) {
    ds4_tokenize_text(e, content, tokens);
}

void ds4_chat_append_assistant_prefix(ds4_engine *e, ds4_tokens *tokens,
                                      ds4_think_mode mode) {
    (void)e; (void)tokens; (void)mode;
}

char *ds4_token_text(ds4_engine *e, int token, size_t *len) {
    (void)e;
    static char buf[2];
    buf[0] = (char)token;
    if (len) *len = 1;
    return buf;
}

int ds4_token_eos(ds4_engine *e) { (void)e; return 0; }
int ds4_token_user(ds4_engine *e) { (void)e; return 1; }
int ds4_token_assistant(ds4_engine *e) { (void)e; return 2; }

const ds4_tokens *ds4_session_tokens(ds4_session *s) { return &s->tokens; }

/* ========================================================================
 * Session management
 * ======================================================================== */

int ds4_session_create(ds4_session **out, ds4_engine *e, int ctx_size) {
    ds4_session *s = (ds4_session *)xcalloc(1, sizeof(ds4_session));
    s->engine = e;
    s->ctx_size = ctx_size;
    s->pos = 0;
    s->logits = (float *)xcalloc((size_t)e->shape.n_vocab, sizeof(float));

    /* Create KV cache */
    s->kv_cache = ds4_kv_cache_create(
        ctx_size, e->shape.n_layer, e->shape.n_head_kv, e->shape.n_head_dim);

    /* Allocate GPU scratch buffer for layer forward */
    uint32_t max_scratch = 4096 + 32768 + 512 + 32768 + 8192 + 256; /* H + QD + KVD + tmp_out + O_OUT + NE */
    s->d_scratch = NULL;
    if (cudaMalloc(&s->d_scratch, max_scratch * sizeof(float)) != cudaSuccess) { fprintf(stderr, "Failed to allocate scratch\n"); return 1; }
    if (cudaMalloc(&s->d_tokens, ctx_size * sizeof(int32_t)) != cudaSuccess) { fprintf(stderr, "Failed to allocate token buffer\n"); return 1; }

    fprintf(stderr, "[ds4] Session created: ctx=%d, kv_cache=%.2f MiB\n",
            ctx_size, ds4_kv_cache_memory_bytes(s->kv_cache) / (1024.0 * 1024.0));
    *out = s;
    return 0;
}

void ds4_session_free(ds4_session *s) {
    if (!s) return;
    ds4_tokens_free(&s->tokens);
    free(s->logits);
    ds4_kv_cache_free(s->kv_cache);
    if (s->d_scratch) cudaFree(s->d_scratch);
    if (s->d_tokens) cudaFree(s->d_tokens);
    free(s);
}

int ds4_session_power(ds4_session *s) { (void)s; return 100; }
int ds4_session_set_power(ds4_session *s, int p) { (void)s; (void)p; return 0; }
bool ds4_session_is_distributed(ds4_session *s) { (void)s; return false; }

void ds4_session_set_progress(ds4_session *s, ds4_session_progress_fn fn, void *ud) {
    s->progress_fn = fn; s->progress_ud = ud;
}

void ds4_session_set_display_progress(ds4_session *s, ds4_session_progress_fn fn, void *ud) {
    s->progress_fn = fn; s->progress_ud = ud;
}

void ds4_session_set_cancel(ds4_session *s, ds4_session_cancel_fn fn, void *ud) {
    s->cancel_fn = fn; s->cancel_ud = ud;
}

void ds4_session_report_progress(ds4_session *s, const char *event, int current, int total) {
    if (s->progress_fn) s->progress_fn(s->progress_ud, event, current, total);
}

int ds4_session_distributed_route_ready(ds4_session *s, char *err, size_t errlen) {
    (void)s;
    if (err && errlen > 0) err[0] = 0;
    return 1;
}

/* ========================================================================
 * Inference: full forward pass
 * ======================================================================== */

/* Pre-load all layer weights to GPU */
static void preload_all_layers(ds4_engine *e) {
    layer_weights_t *d_layers;
    cudaError_t cerr = cudaMallocManaged(&d_layers, e->shape.n_layer * sizeof(layer_weights_t));
    if (cerr != cudaSuccess) { fprintf(stderr, "cudaMallocManaged d_layers failed: %s\n", cudaGetErrorString(cerr)); fflush(stderr); exit(1); }
    
    for (uint32_t l = 0; l < e->shape.n_layer; l++) {
        layer_weights_t lw;
        load_layer_weights(e->sst, l, &lw);
        cudaMemcpy(&d_layers[l], &lw, sizeof(layer_weights_t), cudaMemcpyHostToDevice);
        free(lw.gate);
    }
    
    e->d_layer_weights = d_layers;
    fprintf(stderr, "[ds4] Pre-loaded %u layers to GPU (%.2f MiB)\n",
            e->shape.n_layer,
            e->shape.n_layer * sizeof(layer_weights_t) / (1024.0 * 1024.0));
}

int ds4_session_sync(ds4_session *s, const ds4_tokens *prompt, char *err, size_t errlen) {
    ds4_tokens_free(&s->tokens);
    ds4_tokens_copy(&s->tokens, prompt);
    s->pos = s->tokens.len;

    if (s->tokens.len == 0) { if (err && errlen > 0) err[0] = 0; return 0; }

    /* Reset KV cache for new sequence */
    ds4_kv_cache_reset(s->kv_cache);

    /* Embed all tokens (batch embedding for prefill) */

    fprintf(stderr, "[DEBUG] Launching token embedding...\n"); fflush(stderr);
    cudaMemcpy(s->d_tokens, s->tokens.v, s->tokens.len * sizeof(int32_t), cudaMemcpyHostToDevice);
    launch_token_embedding(s->engine->d_embedding, s->d_tokens,
                           s->d_scratch, s->engine->shape.n_vocab,
                           s->engine->shape.n_embd, s->tokens.len);
    cudaDeviceSynchronize();
    { cudaError_t e = cudaGetLastError(); if (e != cudaSuccess) { fprintf(stderr, "embedding error: %s\n", cudaGetErrorString(e)); fflush(stderr); exit(1); } }

    /* Run all 43 layers */
    for (uint32_t l = 0; l < s->engine->shape.n_layer; l++) {
        layer_weights_t lw;
        load_layer_weights(s->engine->sst, l, &lw);
        
        void *d_lw;
        cudaError_t cerr = cudaMallocManaged(&d_lw, sizeof(layer_weights_t));
        if (cerr != cudaSuccess) { fprintf(stderr, "cudaMallocManaged d_lw failed: %s\n", cudaGetErrorString(cerr)); fflush(stderr); exit(1); }
        cudaMemcpy(d_lw, &lw, sizeof(layer_weights_t), cudaMemcpyHostToDevice);
        
        float *d_in = (l == 0) ? s->d_scratch : s->d_scratch + 4096;
        float *d_out = (l == s->engine->shape.n_layer - 1) ? s->d_scratch + 4096 : s->d_scratch;
        float *scratch = s->d_scratch + 8192;
        
        fprintf(stderr, "  d_lw=%p d_in=%p d_out=%p scratch=%p pos=%d\n", d_lw, d_in, d_out, scratch, s->pos - 1); fflush(stderr);
        fprintf(stderr, "[DEBUG] Entered ds4_layer_forward layer %u\n", l); fflush(stderr);
        ds4_layer_forward(d_lw, s->kv_cache, l, d_in, d_out, s->pos - 1,
                          scratch, s->engine->shape.rms_eps,
                          s->engine->shape.expert_weight_scale,
                          s->engine->shape.rope_freq_base);
        
        cudaFree(d_lw);
    }

    /* Output projection: hidden → logits */

    fprintf(stderr, "[DEBUG] Launching output projection...\n");
    launch_output_projection(s->d_scratch + 4096, s->engine->d_output_w,
                             s->logits, s->engine->shape.n_vocab,
                             s->engine->shape.n_embd);

    if (err && errlen > 0) err[0] = 0;
    return 0;
}

bool ds4_session_rewrite_requires_rebuild(int live_len, int canonical_len, int common) {
    return live_len != canonical_len || canonical_len != common;
}

ds4_session_rewrite_result ds4_session_rewrite_from_common(
        ds4_session *s, const ds4_tokens *prompt, int common,
        char *err, size_t errlen) {
    (void)common;
    return ds4_session_sync(s, prompt, err, errlen) == 0
        ? DS4_SESSION_REWRITE_OK : DS4_SESSION_REWRITE_ERROR;
}

int ds4_session_common_prefix(ds4_session *s, const ds4_tokens *prompt) {
    int common = 0;
    int max = s->tokens.len < prompt->len ? s->tokens.len : prompt->len;
    while (common < max && s->tokens.v[common] == prompt->v[common]) common++;
    return common;
}

int ds4_session_argmax(ds4_session *s) {
    int best = 0;
    for (int i = 1; i < s->engine->shape.n_vocab; i++) {
        if (s->logits[i] > s->logits[best]) best = i;
    }
    return best;
}

int ds4_session_argmax_excluding(ds4_session *s, int excluded) {
    int best = -1;
    for (int i = 0; i < s->engine->shape.n_vocab; i++) {
        if (i == excluded) continue;
        if (best < 0 || s->logits[i] > s->logits[best]) best = i;
    }
    return best >= 0 ? best : 0;
}

int ds4_sample_logits(const float *logits, int n_vocab, float temperature,
                      int top_k, float top_p, float min_p, uint64_t *rng) {
    (void)top_k; (void)top_p; (void)min_p; (void)rng;
    if (temperature <= 0.0f) {
        int best = 0;
        for (int i = 1; i < n_vocab; i++) {
            if (logits[i] > logits[best]) best = i;
        }
        return best;
    }
    int best = 0;
    for (int i = 1; i < n_vocab; i++) {
        if (logits[i] > logits[best]) best = i;
    }
    return best;
}

int ds4_session_sample(ds4_session *s, float temperature, int top_k,
                       float top_p, float min_p, uint64_t *rng) {
    return ds4_sample_logits(s->logits, s->engine->shape.n_vocab,
                             temperature, top_k, top_p, min_p, rng);
}

int ds4_session_top_logprobs(ds4_session *s, ds4_token_score *out, int k) {
    for (int i = 0; i < k && i < s->engine->shape.n_vocab; i++) {
        out[i].id = i; out[i].logit = s->logits[i]; out[i].logprob = 0.0f;
    }
    return k;
}

int ds4_session_token_logprob(ds4_session *s, int token, ds4_token_score *out) {
    if (token < 0 || token >= s->engine->shape.n_vocab) return -1;
    out->id = token; out->logit = s->logits[token]; out->logprob = 0.0f;
    return 0;
}

int ds4_session_copy_logits(ds4_session *s, float *out, int cap) {
    int n = s->engine->shape.n_vocab < cap ? s->engine->shape.n_vocab : cap;
    memcpy(out, s->logits, (size_t)n * sizeof(float));
    return n;
}

int ds4_session_set_logits(ds4_session *s, const float *logits, int n) {
    int copy = n < s->engine->shape.n_vocab ? n : s->engine->shape.n_vocab;
    memcpy(s->logits, logits, (size_t)copy * sizeof(float));
    return 0;
}

int ds4_session_eval(ds4_session *s, int token, char *err, size_t errlen) {
    /* Append token and run forward pass for one position */
    ds4_tokens_push(&s->tokens, token);

    /* Embed token */
    cudaMemcpy(s->d_tokens, &s->tokens.v[s->pos], 1 * sizeof(int32_t), cudaMemcpyHostToDevice);
    launch_token_embedding(s->engine->d_embedding, s->d_tokens,
                           s->d_scratch, s->engine->shape.n_vocab,
                           s->engine->shape.n_embd, 1);

    /* Run all 43 layers */
    for (uint32_t l = 0; l < s->engine->shape.n_layer; l++) {
        layer_weights_t lw;
        load_layer_weights(s->engine->sst, l, &lw);

        void *d_lw;
        cudaError_t cerr = cudaMallocManaged(&d_lw, sizeof(layer_weights_t));
        if (cerr != cudaSuccess) { fprintf(stderr, "cudaMallocManaged d_lw failed: %s\n", cudaGetErrorString(cerr)); fflush(stderr); exit(1); }
        cudaMemcpy(d_lw, &lw, sizeof(layer_weights_t), cudaMemcpyHostToDevice);

        float *d_out = (l == s->engine->shape.n_layer - 1) ? s->d_scratch : s->d_scratch;
        float *d_in = (l == 0) ? s->d_scratch : s->d_scratch;

        fprintf(stderr, "[DEBUG] Entered ds4_layer_forward layer %u\n", l); fflush(stderr);
        ds4_layer_forward(d_lw, s->kv_cache, l, d_in, d_out, s->pos,
                          s->d_scratch + 4096, s->engine->shape.rms_eps,
                          s->engine->shape.expert_weight_scale,
                          s->engine->shape.rope_freq_base);

        cudaFree(d_lw);
        free(lw.gate);
    }

    s->pos++;
    launch_output_projection(s->d_scratch, s->engine->d_output_w,
                             s->logits, s->engine->shape.n_vocab,
                             s->engine->shape.n_embd);

    if (err && errlen > 0) err[0] = 0;
    return 0;
}

int ds4_session_eval_speculative_argmax(ds4_session *s, int first_token,
                                        int max_tokens, int eos_token,
                                        int *accepted, int accepted_cap,
                                        char *err, size_t errlen) {
    (void)max_tokens; (void)eos_token; (void)accepted; (void)accepted_cap;
    return ds4_session_eval(s, first_token, err, errlen);
}

void ds4_session_invalidate(ds4_session *s) {
    ds4_tokens_free(&s->tokens);
    s->pos = 0;
}

void ds4_session_rewind(ds4_session *s, int pos) {
    if (pos < s->tokens.len) {
        s->tokens.len = pos;
        s->pos = pos;
    }
}

int ds4_session_pos(ds4_session *s) { return s->pos; }
int ds4_session_ctx(ds4_session *s) { return s->ctx_size; }
int ds4_session_prefill_cap(ds4_session *s) { return 4096; }

/* ========================================================================
 * Distributed stubs
 * ======================================================================== */

int ds4_session_layer_slice_reset(ds4_session *s, char *err, size_t errlen) {
    (void)s;
    if (err && errlen > 0) err[0] = 0;
    return 0;
}

int ds4_session_eval_layer_slice(ds4_session *s, const int *tokens,
                                 uint32_t n_tokens, uint32_t pos0,
                                 uint32_t layer_start, uint32_t layer_end,
                                 const float *input_hc, float *output_hc,
                                 bool output_logits, float *logits,
                                 char *err, size_t errlen) {
    (void)s; (void)tokens; (void)n_tokens; (void)pos0;
    (void)layer_start; (void)layer_end;
    (void)input_hc; (void)output_hc;
    (void)output_logits; (void)logits;
    if (err && errlen > 0) { snprintf(err, errlen, "distributed not supported"); }
    return -1;
}

int ds4_session_eval_output_head_from_hc(ds4_session *s,
                                         const float *hidden_hc,
                                         uint32_t n_tokens,
                                         float *logits,
                                         char *err, size_t errlen) {
    (void)s; (void)hidden_hc; (void)n_tokens; (void)logits;
    if (err && errlen > 0) { snprintf(err, errlen, "output head not implemented"); }
    return -1;
}

/* ========================================================================
 * Payload / snapshot stubs
 * ======================================================================== */

uint64_t ds4_session_payload_bytes(ds4_session *s) { (void)s; return 0; }

int ds4_session_stage_payload(ds4_session *s, ds4_session_payload_file *out,
                              char *err, size_t errlen) {
    (void)s; (void)out;
    if (err && errlen > 0) snprintf(err, errlen, "payload staging not implemented");
    return -1;
}

int ds4_session_write_staged_payload(const ds4_session_payload_file *payload,
                                     FILE *fp, char *err, size_t errlen) {
    (void)payload; (void)fp;
    if (err && errlen > 0) snprintf(err, errlen, "payload write not implemented");
    return -1;
}

void ds4_session_payload_file_free(ds4_session_payload_file *payload) {
    free(payload->path);
    payload->path = NULL;
}

int ds4_session_save_payload(ds4_session *s, FILE *fp, char *err, size_t errlen) {
    (void)s; (void)fp;
    if (err && errlen > 0) snprintf(err, errlen, "payload save not implemented");
    return -1;
}

int ds4_session_load_payload(ds4_session *s, FILE *fp, uint64_t payload_bytes,
                             char *err, size_t errlen) {
    (void)s; (void)fp; (void)payload_bytes;
    if (err && errlen > 0) snprintf(err, errlen, "payload load not implemented");
    return -1;
}

int ds4_session_save_snapshot(ds4_session *s, ds4_session_snapshot *snap,
                              char *err, size_t errlen) {
    (void)s; (void)snap;
    if (err && errlen > 0) snprintf(err, errlen, "snapshot not implemented");
    return -1;
}

int ds4_session_load_snapshot(ds4_session *s, const ds4_session_snapshot *snap,
                              char *err, size_t errlen) {
    (void)s; (void)snap;
    if (err && errlen > 0) snprintf(err, errlen, "snapshot not implemented");
    return -1;
}

void ds4_session_snapshot_free(ds4_session_snapshot *snap) {
    free(snap->ptr);
    snap->ptr = NULL;
    snap->len = snap->cap = 0;
}

uint64_t ds4_session_layer_payload_bytes(ds4_session *s,
                                         uint32_t layer_start, uint32_t layer_end) {
    (void)s; (void)layer_start; (void)layer_end;
    return 0;
}

int ds4_session_save_layer_payload(ds4_session *s, FILE *fp,
                                   uint32_t layer_start, uint32_t layer_end,
                                   char *err, size_t errlen) {
    (void)s; (void)fp; (void)layer_start; (void)layer_end;
    if (err && errlen > 0) snprintf(err, errlen, "layer payload not implemented");
    return -1;
}

int ds4_session_load_layer_payload(ds4_session *s, FILE *fp,
                                   uint64_t payload_bytes,
                                   const int *tokens, uint32_t n_tokens,
                                   uint32_t layer_start, uint32_t layer_end,
                                   char *err, size_t errlen) {
    (void)s; (void)fp; (void)payload_bytes;
    (void)tokens; (void)n_tokens; (void)layer_start; (void)layer_end;
    if (err && errlen > 0) snprintf(err, errlen, "layer payload not implemented");
    return -1;
}

/* ========================================================================
 * Head / first-token tests (stubs)
 * ======================================================================== */

int ds4_engine_head_test(ds4_engine *e, const ds4_tokens *prompt) {
    (void)e; (void)prompt;
    fprintf(stderr, "[ds4] head_test: not implemented for NVFP4\n");
    return -1;
}

int ds4_engine_first_token_test(ds4_engine *e, const ds4_tokens *prompt) {
    (void)e; (void)prompt;
    fprintf(stderr, "[ds4] first_token_test: not implemented for NVFP4\n");
    return -1;
}

int ds4_engine_metal_graph_test(ds4_engine *e, const ds4_tokens *prompt) {
    (void)e; (void)prompt;
    return -1;
}

int ds4_engine_metal_graph_full_test(ds4_engine *e, const ds4_tokens *prompt) {
    (void)e; (void)prompt;
    return -1;
}

int ds4_engine_metal_graph_prompt_test(ds4_engine *e, const ds4_tokens *prompt,
                                       int ctx_size) {
    (void)e; (void)prompt; (void)ctx_size;
    return -1;
}

void ds4_engine_dump_tokens(ds4_engine *e, const ds4_tokens *tokens) {
    (void)e;
    for (int i = 0; i < tokens->len; i++) {
        fprintf(stderr, "%d ", tokens->v[i]);
    }
    fprintf(stderr, "\n");
}

int ds4_dump_text_tokenization(const char *model_path, const char *text, FILE *fp) {
    (void)model_path; (void)text; (void)fp;
    return -1;
}

/* ========================================================================
 * Imatrix (stub)
 * ======================================================================== */

int ds4_engine_collect_imatrix(ds4_engine *s, const char *dataset_path,
                               const char *output_path, int ctx_size,
                               int max_prompts, int max_tokens) {
    (void)s; (void)dataset_path; (void)output_path;
    (void)ctx_size; (void)max_prompts; (void)max_tokens;
    fprintf(stderr, "[ds4] imatrix: not supported for NVFP4\n");
    return -1;
}

/* ========================================================================
 * GPU hooks
 * ======================================================================== */

void ds4_gpu_print_memory_report(const char *label) {
    (void)label;
    fprintf(stderr, "[GPU] %s: CUDA runtime available via nvcc builds\n", label ? label : "memory");
}

/* ========================================================================
 * Main entry point
 * ======================================================================== */

int main(int argc, char **argv) {
    fprintf(stderr, "ds4-nvfp4 v1.0 (sm_121)\n");
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <model_dir>\n", argv[0]);
        return 1;
    }
    ds4_engine *engine = NULL;
    ds4_engine_options opt = {0};
    opt.model_path = argv[1];
    opt.backend = DS4_BACKEND_CUDA;
    if (ds4_engine_open(&engine, &opt) != 0) {
        fprintf(stderr, "Failed to load model\n");
        return 1;
    }
    ds4_engine_summary(engine);

    /* Create a session and run a test prompt */
    ds4_session *session = NULL;
    int ctx_size = 4096;
    if (ds4_session_create(&session, engine, ctx_size) != 0) {
        fprintf(stderr, "Failed to create session\n");
        ds4_engine_close(engine);
        return 1;
    }

    ds4_tokens prompt = {0};
    ds4_tokenize_text(engine, "Hello, world!", &prompt);

    char err[256];
    ds4_session_sync(session, &prompt, err, sizeof(err));
    fprintf(stderr, "Sync result: %s\n", err[0] ? err : "OK");

    int token = ds4_session_argmax(session);
    fprintf(stderr, "First token: %s\n", err[0] ? err : "OK");

    ds4_session_free(session);
    ds4_engine_close(engine);
    return 0;
}
