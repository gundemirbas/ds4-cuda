/**
 * ds4 Model Configuration Parser
 *
 * Parses DeepSeek-V4-Flash config.json into a ds4_model_config struct.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>
#include <ctype.h>

#include "ds4_config.h"

/* Simple JSON value parser (string, number, object, array) */
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

static char *json_parse_string(json_ctx *c) {
    json_skip_ws(c);
    if (c->p >= c->end || *c->p != '"') return NULL;
    c->p++; /* skip opening quote */
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
    c->p++; /* skip closing quote */
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

static void json_skip_value(json_ctx *c) {
    json_skip_ws(c);
    if (c->p >= c->end) return;
    char ch = *c->p;
    if (ch == '"') { json_parse_string(c); return; }
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
    /* skip number/true/false/null */
    while (c->p < c->end && *c->p != ',' && *c->p != '}' && *c->p != ']' && *c->p != ' ')
        c->p++;
}

/* ========================================================================
 * Main config parser
 * ======================================================================== */

ds4_model_config *ds4_config_load(const char *path) {
    FILE *fp = fopen(path, "rb");
    if (!fp) {
        fprintf(stderr, "open config %s: %s\n", path, strerror(errno));
        return NULL;
    }

    fseek(fp, 0, SEEK_END);
    long fsize = ftell(fp);
    fseek(fp, 0, SEEK_SET);
    if (fsize <= 0) { fclose(fp); return NULL; }

    char *buf = malloc(fsize + 1);
    if (!buf) { fclose(fp); return NULL; }
    size_t nread = fread(buf, 1, fsize, fp);
    fclose(fp);
    buf[nread] = '\0';

    json_ctx ctx;
    ctx.p = buf;
    ctx.end = buf + nread;

    ds4_model_config *cfg = calloc(1, sizeof(ds4_model_config));
    if (!cfg) { free(buf); return NULL; }

    if (json_expect(&ctx, '{') < 0) { free(buf); free(cfg); return NULL; }

    while (ctx.p < ctx.end) {
        json_skip_ws(&ctx);
        if (*ctx.p == '}') break;
        if (*ctx.p == ',') { ctx.p++; continue; }

        char *key = json_parse_string(&ctx);
        if (!key) break;
        json_skip_ws(&ctx);
        if (ctx.p >= ctx.end || *ctx.p != ':') { free(key); break; }
        ctx.p++; /* skip : */

        if (strcmp(key, "n_layers") == 0)           cfg->n_layers       = (int)json_parse_int(&ctx);
        else if (strcmp(key, "n_heads") == 0)        cfg->n_heads        = (int)json_parse_int(&ctx);
        else if (strcmp(key, "n_kv_heads") == 0)     cfg->n_kv_heads     = (int)json_parse_int(&ctx);
        else if (strcmp(key, "head_dim") == 0)       cfg->head_dim       = (int)json_parse_int(&ctx);
        else if (strcmp(key, "n_experts") == 0)      cfg->n_experts      = (int)json_parse_int(&ctx);
        else if (strcmp(key, "n_experts_per_layer") == 0) cfg->n_experts  = (int)json_parse_int(&ctx);
        else if (strcmp(key, "n_activated_experts") == 0) cfg->n_activated_experts = (int)json_parse_int(&ctx);
        else if (strcmp(key, "n_activated_experts_per_layer") == 0) cfg->n_activated_experts = (int)json_parse_int(&ctx);
        else if (strcmp(key, "vocab_size") == 0)     cfg->vocab_size     = (int)json_parse_int(&ctx);
        else if (strcmp(key, "hidden_size") == 0)    cfg->hidden_size    = (int)json_parse_int(&ctx);
        else if (strcmp(key, "intermediate_size") == 0) cfg->intermediate_size = (int)json_parse_int(&ctx);
        else if (strcmp(key, "n_mtp_layers") == 0)   cfg->n_mtp_layers   = (int)json_parse_int(&ctx);
        else if (strcmp(key, "max_seq_len") == 0)    cfg->max_seq_len    = (int)json_parse_int(&ctx);
        else if (strcmp(key, "rope_dim") == 0)       cfg->rope_dim       = (int)json_parse_int(&ctx);
        else if (strcmp(key, "rope_freq_base") == 0) cfg->rope_freq_base = json_parse_int(&ctx);
        else if (strcmp(key, "quantization") == 0)   cfg->quantization   = json_parse_string(&ctx);
        else json_skip_value(&ctx);

        free(key);
    }

    /* Fill defaults for missing fields */
    if (cfg->n_activated_experts == 0) cfg->n_activated_experts = 6;
    if (cfg->head_dim == 0) cfg->head_dim = 512;
    if (cfg->rope_dim == 0) cfg->rope_dim = cfg->head_dim;
    if (cfg->n_kv_heads == 0) cfg->n_kv_heads = 1;
    if (cfg->rope_freq_base == 0) cfg->rope_freq_base = 10000;

    free(buf);
    return cfg;
}

void ds4_config_free(ds4_model_config *cfg) {
    if (!cfg) return;
    free(cfg->quantization);
    free(cfg);
}
