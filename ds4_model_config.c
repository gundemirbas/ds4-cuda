/**
 * ds4_model_config.c — config.json parser for DeepSeek-V4-Flash-NVFP4.
 *
 * Reads config.json from a model directory and populates a ds4_config struct
 * with model shape parameters needed by the inference engine.
 */

#include "ds4_config.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>
#include <ctype.h>
#include <sys/stat.h>

/* ---- Simple JSON parser ---- */

typedef struct {
    const char *p;
    const char *end;
} json_ctx;

static void json_skip_ws(json_ctx *c) {
    while (c->p < c->end && (*c->p == ' ' || *c->p == '\t' || *c->p == '\n' || *c->p == '\r'))
        c->p++;
}

static int json_expect(json_ctx *c, char expected) {
    json_skip_ws(c);
    if (c->p >= c->end || *c->p != expected) return -1;
    c->p++;
    return 0;
}

static char *cfg_json_parse_string(json_ctx *c) {
    json_skip_ws(c);
    if (c->p >= c->end || *c->p != '"') return NULL;
    c->p++;
    const char *start = c->p;
    while (c->p < c->end && *c->p != '"') {
        if (*c->p == '\\') c->p++;
        c->p++;
    }
    if (c->p >= c->end) return NULL;
    size_t len = c->p - start;
    char *s = malloc(len + 1);
    if (!s) return NULL;
    memcpy(s, start, len);
    s[len] = '\0';
    c->p++;
    return s;
}

static int64_t json_parse_int(json_ctx *c) {
    json_skip_ws(c);
    int sign = 1;
    if (c->p < c->end && *c->p == '-') { sign = -1; c->p++; }
    int64_t v = 0;
    while (c->p < c->end && isdigit(*c->p)) {
        v = v * 10 + (*c->p - '0');
        c->p++;
    }
    return v * sign;
}

static void cfg_json_skip_value(json_ctx *c) {
    json_skip_ws(c);
    if (c->p >= c->end) return;
    char ch = *c->p;
    if (ch == '"') { cfg_json_parse_string(c); return; }
    if (ch == '{') {
        c->p++;
        int depth = 1;
        while (depth > 0 && c->p < c->end) {
            ch = *c->p++;
            if (ch == '{') depth++;
            else if (ch == '}') depth--;
        }
        return;
    }
    if (ch == '[') {
        c->p++;
        int depth = 1;
        while (depth > 0 && c->p < c->end) {
            ch = *c->p++;
            if (ch == '[') depth++;
            else if (ch == ']') depth--;
        }
        return;
    }
    while (c->p < c->end && *c->p != ',' && *c->p != '}' && *c->p != ']' && *c->p != ' ')
        c->p++;
}

/* ---- Read file into buffer ---- */

static char *read_file(const char *path, size_t *len) {
    FILE *fp = fopen(path, "rb");
    if (!fp) return NULL;

    struct stat st;
    if (fstat(fileno(fp), &st) == -1 || st.st_size <= 0) {
        fclose(fp);
        return NULL;
    }

    char *buf = malloc((size_t)st.st_size + 1);
    if (!buf) { fclose(fp); return NULL; }

    size_t nread = fread(buf, 1, (size_t)st.st_size, fp);
    fclose(fp);
    buf[nread] = '\0';
    *len = nread;
    return buf;
}

/* ---- Build config.json path ---- */

static char *config_path(const char *model_dir) {
    size_t dirlen = strlen(model_dir);
    /* strip trailing slash */
    while (dirlen > 0 && model_dir[dirlen - 1] == '/') dirlen--;
    size_t pathlen = dirlen + 12; /* "/config.json\0" */
    char *path = malloc(pathlen);
    if (!path) return NULL;
    memcpy(path, model_dir, dirlen);
    memcpy(path + dirlen, "/config.json", 12);
    return path;
}

/* ---- Parse config.json into ds4_config ---- */

static void set_string(const char **dst, char *val) {
    if (*dst) free((void*)*dst);
    *dst = val ? val : NULL;
}

int ds4_config_load(const char *model_dir, ds4_config *cfg) {
    memset(cfg, 0, sizeof(*cfg));

    /* Build config.json path */
    char *path = config_path(model_dir);
    if (!path) return 0;

    /* Read file */
    size_t file_len;
    char *buf = read_file(path, &file_len);
    free(path);
    if (!buf) return 0;

    /* Parse JSON */
    json_ctx ctx;
    ctx.p = buf;
    ctx.end = buf + file_len;

    if (json_expect(&ctx, '{') < 0) { free(buf); return 0; }

    /* Defaults */
    cfg->n_expert_used = 6;
    cfg->head_dim = 512;
    cfg->n_kv_heads = 1;
    cfg->rms_eps = 1.0e-6f;
    cfg->hc_eps = 1.0e-6f;
    cfg->swiglu_clamp_exp = 10.0f;
    cfg->rope_freq_base = 10000.0f;
    cfg->rope_scale_factor = 16.0f;
    cfg->rope_yarn_beta_fast = 32.0f;
    cfg->rope_yarn_beta_slow = 1.0f;
    cfg->compress_rope_freq_base = 160000.0f;
    cfg->rope_orig_ctx = 65536;
    cfg->expert_weight_scale = 1.0f;

    while (ctx.p < ctx.end) {
        json_skip_ws(&ctx);
        if (*ctx.p == '}') break;
        if (*ctx.p == ',') { ctx.p++; continue; }

        char *key = cfg_json_parse_string(&ctx);
        if (!key) break;
        json_skip_ws(&ctx);
        if (ctx.p >= ctx.end || *ctx.p != ':') { free(key); break; }
        ctx.p++;

        /* Model name */
        if (strcmp(key, "_name_or_path") == 0 ||
            strcmp(key, "name") == 0)
            set_string(&cfg->name, cfg_json_parse_string(&ctx));

        /* Architecture */
        else if (strcmp(key, "n_layers") == 0 ||
                 strcmp(key, "num_hidden_layers") == 0)
            cfg->n_layer = (uint32_t)json_parse_int(&ctx);
        else if (strcmp(key, "n_heads") == 0 ||
                 strcmp(key, "num_attention_heads") == 0)
            cfg->n_head = (uint32_t)json_parse_int(&ctx);
        else if (strcmp(key, "n_kv_heads") == 0 ||
                 strcmp(key, "num_key_value_heads") == 0)
            cfg->n_kv_heads = (uint32_t)json_parse_int(&ctx);
        else if (strcmp(key, "head_dim") == 0)
            cfg->head_dim = (uint32_t)json_parse_int(&ctx);
        else if (strcmp(key, "hidden_size") == 0 ||
                 strcmp(key, "embedding_length") == 0 ||
                 strcmp(key, "d_model") == 0)
            cfg->n_embd = (uint32_t)json_parse_int(&ctx);
        else if (strcmp(key, "vocab_size") == 0)
            cfg->n_vocab = (uint32_t)json_parse_int(&ctx);
        else if (strcmp(key, "n_experts") == 0 ||
                 strcmp(key, "num_experts") == 0 ||
                 strcmp(key, "num_experts_per_layer") == 0)
            cfg->n_expert = (uint32_t)json_parse_int(&ctx);
        else if (strcmp(key, "n_experts_per_layer") == 0)
            cfg->n_expert = (uint32_t)json_parse_int(&ctx);
        else if (strcmp(key, "n_activated_experts") == 0 ||
                 strcmp(key, "n_activated_experts_per_layer") == 0)
            cfg->n_expert_used = (uint32_t)json_parse_int(&ctx);
        else if (strcmp(key, "intermediate_size") == 0 ||
                 strcmp(key, "ffn_hidden_size") == 0)
            cfg->n_ff_exp = (uint32_t)json_parse_int(&ctx);
        else if (strcmp(key, "max_seq_len") == 0 ||
                 strcmp(key, "max_position_embeddings") == 0)
            cfg->max_seq_len = (uint64_t)json_parse_int(&ctx);
        else if (strcmp(key, "rope_freq_base") == 0 ||
                 strcmp(key, "rope_theta") == 0)
            cfg->rope_freq_base = (float)json_parse_int(&ctx);
        else if (strcmp(key, "rope_scaling_factor") == 0 ||
                 strcmp(key, "rope_factor") == 0)
            cfg->rope_scale_factor = (float)json_parse_int(&ctx);

        /* DeepSeek-specific */
        else if (strcmp(key, "num_expert_shared") == 0 ||
                 strcmp(key, "n_expert_shared") == 0)
            cfg->n_expert_shared = (uint32_t)json_parse_int(&ctx);
        else if (strcmp(key, "num_expert_groups") == 0)
            cfg->n_expert_groups = (uint32_t)json_parse_int(&ctx);
        else if (strcmp(key, "num_group_used") == 0)
            cfg->n_group_used = (uint32_t)json_parse_int(&ctx);
        else if (strcmp(key, "num_hash_layer") == 0 ||
                 strcmp(key, "n_hash_layer") == 0)
            cfg->n_hash_layer = (uint32_t)json_parse_int(&ctx);
        else if (strcmp(key, "sliding_window") == 0)
            cfg->n_swa = (uint32_t)json_parse_int(&ctx);
        else if (strcmp(key, "n_indexer_heads") == 0)
            cfg->n_indexer_heads = (uint32_t)json_parse_int(&ctx);
        else if (strcmp(key, "n_indexer_head_dim") == 0)
            cfg->n_indexer_head_dim = (uint32_t)json_parse_int(&ctx);
        else if (strcmp(key, "n_indexer_top_k") == 0)
            cfg->n_indexer_top_k = (uint32_t)json_parse_int(&ctx);
        else if (strcmp(key, "n_lora_q") == 0)
            cfg->n_lora_q = (uint32_t)json_parse_int(&ctx);
        else if (strcmp(key, "n_lora_o") == 0)
            cfg->n_lora_o = (uint32_t)json_parse_int(&ctx);
        else if (strcmp(key, "expert_weight_scale") == 0)
            cfg->expert_weight_scale = (float)json_parse_int(&ctx);
        else if (strcmp(key, "rms_norm_eps") == 0)
            cfg->rms_eps = (float)json_parse_int(&ctx);
        else if (strcmp(key, "rope_orig_ctx") == 0)
            cfg->rope_orig_ctx = (uint64_t)json_parse_int(&ctx);

        cfg_json_skip_value(&ctx);
        free(key);
    }

    /* Derived fields */
    /* n_embd is the canonical field - hidden_size removed */
    if (cfg->n_swa == 0) cfg->n_swa = cfg->max_seq_len > 0 ? (uint32_t)cfg->max_seq_len : 4096;

    free(buf);
    return 1;
}
