/**
 * ds4 Safetensors Parser
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <errno.h>
#include <ctype.h>
#include <math.h>
#include <dirent.h>

#include "ds4_safetensors.h"

/* ========================================================================
 * JSON helpers
 * ======================================================================== */

static char json_peek(const char **p, const char *end) {
    while (*p < end && (**p == ' ' || **p == '\t' || **p == '\n' || **p == '\r'))
        (*p)++;
    return (*p < end) ? **p : '\0';
}

static char json_advance(const char **p, const char *end) {
    while (*p < end && (**p == ' ' || **p == '\t' || **p == '\n' || **p == '\r'))
        (*p)++;
    return (*p < end) ? *((*p)++) : '\0';
}

static char *json_parse_string(const char **p, const char *end) {
    if (json_advance(p, end) != '"') return NULL;
    const char *start = *p;
    while (*p < end && **p != '"') {
        if (**p == '\\') (*p)++;
        if (*p < end) (*p)++;
    }
    if (*p >= end) return NULL;
    size_t len = *p - start;
    char *str = malloc(len + 1);
    if (!str) return NULL;
    memcpy(str, start, len);
    str[len] = '\0';
    (*p)++;
    return str;
}

static void json_skip_value(const char **p, const char *end) {
    char c = json_peek(p, end);
    if (c == '"') {
        json_parse_string(p, end);
    } else if (c == '{' || c == '[') {
        char open = c;
        char close = (c == '{') ? '}' : ']';
        (*p)++;
        int depth = 1;
        while (depth > 0 && *p < end) {
            c = *((*p)++);
            if (c == open) depth++;
            else if (c == close) depth--;
        }
    } else {
        while (*p < end && **p != ',' && **p != '}' && **p != ']')
            (*p)++;
    }
}

static double json_parse_number(const char **p, const char *end) {
    while (*p < end && (**p == ' ' || **p == '\t')) (*p)++;
    const char *start = *p;
    while (*p < end && (**p == '-' || **p == '.' || **p == 'e' || **p == 'E' || isdigit(**p)))
        (*p)++;
    size_t len = *p - start;
    if (len == 0) return 0;
    char buf[64];
    if (len >= sizeof(buf)) len = sizeof(buf) - 1;
    memcpy(buf, start, len);
    buf[len] = '\0';
    return atof(buf);
}

/* ========================================================================
 * Dtype helpers
 * ======================================================================== */

static sst_dtype parse_dtype(const char *s) {
    if (!s) return SST_DTYPE_UNKNOWN;
    if (strcmp(s, "F32") == 0 || strcmp(s, "float32") == 0)    return SST_DTYPE_F32;
    if (strcmp(s, "F16") == 0 || strcmp(s, "float16") == 0)    return SST_DTYPE_F16;
    if (strcmp(s, "BF16") == 0 || strcmp(s, "bfloat16") == 0)  return SST_DTYPE_BF16;
    if (strcmp(s, "F8_E4M3") == 0 || strcmp(s, "fp8") == 0)   return SST_DTYPE_F8_E4M3;
    if (strcmp(s, "NVFP4") == 0 || strcmp(s, "nve4") == 0)    return SST_DTYPE_NVFP4;
    if (strcmp(s, "MXFP4") == 0 || strcmp(s, "mxfp4") == 0)   return SST_DTYPE_MXFP4;
    return SST_DTYPE_UNKNOWN;
}

size_t sst_dtype_size(sst_dtype d) {
    switch (d) {
        case SST_DTYPE_F32:      return 4;
        case SST_DTYPE_F16:      return 2;
        case SST_DTYPE_BF16:     return 2;
        case SST_DTYPE_F8_E4M3:  return 1;
        case SST_DTYPE_NVFP4:    return 1;
        case SST_DTYPE_MXFP4:    return 1;
        default:                 return 0;
    }
}

uint32_t sst_dtype_block_size(sst_dtype d) {
    switch (d) {
        case SST_DTYPE_F32:      return 1;
        case SST_DTYPE_F16:      return 1;
        case SST_DTYPE_BF16:     return 1;
        case SST_DTYPE_F8_E4M3:  return 128;
        case SST_DTYPE_NVFP4:    return 16;
        case SST_DTYPE_MXFP4:    return 32;
        default:                 return 1;
    }
}

uint32_t sst_dtype_bits(sst_dtype d) {
    switch (d) {
        case SST_DTYPE_F32:      return 32;
        case SST_DTYPE_F16:      return 16;
        case SST_DTYPE_BF16:     return 16;
        case SST_DTYPE_F8_E4M3:  return 8;
        case SST_DTYPE_NVFP4:    return 4;
        case SST_DTYPE_MXFP4:    return 4;
        default:                 return 0;
    }
}

/* ========================================================================
 * Tensor helpers
 * ======================================================================== */

uint64_t sst_tensor_elements(const sst_tensor *t) {
    if (!t || t->ndim == 0) return 0;
    uint64_t n = 1;
    for (uint64_t i = 0; i < t->ndim; i++) n *= t->shape[i];
    return n;
}

uint64_t sst_tensor_bytes(const sst_tensor *t) {
    return t ? t->data_size : 0;
}

/* ========================================================================
 * Model loading
 * ======================================================================== */

static int parse_one_tensor(const char **p, const char *end, sst_tensor *out) {
    memset(out, 0, sizeof(*out));

    out->name = json_parse_string(p, end);
    if (!out->name) return -1;
    if (json_advance(p, end) != ':') { free(out->name); out->name = NULL; return -1; }
    if (json_advance(p, end) != '{') { free(out->name); out->name = NULL; return -1; }

    while (*p < end) {
        char c = json_peek(p, end);
        if (c == '}') { (*p)++; break; }
        if (c == ',') { (*p)++; continue; }

        char *key = json_parse_string(p, end);
        if (!key) return -1;
        if (json_advance(p, end) != ':') { free(key); return -1; }

        if (strcmp(key, "dtype") == 0) {
            char *val = json_parse_string(p, end);
            out->dtype = parse_dtype(val);
            free(val);
        } else if (strcmp(key, "shape") == 0) {
            if (json_advance(p, end) != '[') { free(key); return -1; }
            out->ndim = 0;
            while (*p < end) {
                c = json_peek(p, end);
                if (c == ']') { (*p)++; break; }
                if (c == ',') { (*p)++; continue; }
                out->shape[out->ndim++] = (uint64_t)json_parse_number(p, end);
            }
        } else if (strcmp(key, "data_offsets") == 0) {
            if (json_advance(p, end) != '[') { free(key); return -1; }
            uint64_t start = (uint64_t)json_parse_number(p, end);
            if (json_peek(p, end) == ',') (*p)++;
            uint64_t end_off = (uint64_t)json_parse_number(p, end);
            out->data_offset = start;
            out->data_size = end_off - start;
            if (json_advance(p, end) != ']') { free(key); return -1; }
        } else {
            json_skip_value(p, end);
        }
        free(key);
    }
    return 0;
}

sst_model *sst_model_load(const char *path) {
    int fd = open(path, O_RDONLY);
    if (fd < 0) { fprintf(stderr, "open: %s\n", strerror(errno)); return NULL; }

    struct stat st;
    if (fstat(fd, &st) < 0) { close(fd); return NULL; }

    uint64_t file_size = st.st_size;
    void *map = mmap(NULL, file_size, PROT_READ, MAP_PRIVATE, fd, 0);
    if (map == MAP_FAILED) { close(fd); return NULL; }

    sst_model *m = calloc(1, sizeof(sst_model));
    if (!m) { munmap(map, file_size); close(fd); return NULL; }

    m->fd = fd;
    m->map = map;
    m->file_size = file_size;
    m->header_size = *(uint64_t *)m->map;
    m->name = strdup(path);

    const char *json_start = (const char *)m->map + 8;
    const char *json_end = json_start + m->header_size;
    const char *p = json_start;

    if (json_advance(&p, json_end) != '{') { sst_model_free(m); return NULL; }

    /* First pass: count tensors */
    {
        const char *q = p;
        uint64_t count = 0;
        while (q < json_end) {
            char c = json_peek(&q, json_end);
            if (c == '}') break;
            if (c == ',') { q++; continue; }
            char *k = json_parse_string(&q, json_end);
            if (!k) break;
            free(k);
            if (json_advance(&q, json_end) != ':') break;
            json_skip_value(&q, json_end);
            count++;
        }
        m->n_tensors = count;
    }

    m->tensors = calloc(m->n_tensors, sizeof(sst_tensor));
    if (!m->tensors) { sst_model_free(m); return NULL; }

    /* Second pass: parse tensors */
    for (uint64_t i = 0; i < m->n_tensors; i++) {
        char c = json_peek(&p, json_end);
        if (c == ',') p++;
        if (parse_one_tensor(&p, json_end, &m->tensors[i]) < 0) {
            fprintf(stderr, "parse tensor %lu failed\n", i);
            sst_model_free(m);
            return NULL;
        }
    }

    m->data_offset = 8 + m->header_size;
    m->data_size = file_size - m->data_offset;
    return m;
}

void sst_model_free(sst_model *m) {
    if (!m) return;
    if (m->tensors) {
        for (uint64_t i = 0; i < m->n_tensors; i++)
            free(m->tensors[i].name);
        free(m->tensors);
    }
    if (m->map) munmap(m->map, m->file_size);
    if (m->fd >= 0) close(m->fd);
    free(m->name);
    free(m);
}

sst_tensor *sst_model_find_tensor(sst_model *m, const char *name) {
    if (!m || !name) return NULL;
    for (uint64_t i = 0; i < m->n_tensors; i++)
        if (m->tensors[i].name && strcmp(m->tensors[i].name, name) == 0)
            return &m->tensors[i];
    return NULL;
}

void *sst_model_tensor_data(sst_model *m, sst_tensor *t) {
    if (!m || !t) return NULL;
    uint64_t off = m->data_offset + t->data_offset;
    if (off + t->data_size > m->file_size) return NULL;
    return m->map + off;
}

/* ========================================================================
 * Sharded model
 * ======================================================================== */

/* Sort function for qsort - alphabetical order */
static int compare_strings(const void *a, const void *b) {
    const char *sa = *(const char **)a;
    const char *sb = *(const char **)b;
    return strcmp(sa, sb);
}

sst_sharded_model *sst_sharded_model_load(const char *dir_path) {
    DIR *dir = opendir(dir_path);
    if (!dir) {
        fprintf(stderr, "opendir %s: %s\n", dir_path, strerror(errno));
        return NULL;
    }

    /* Collect .safetensors files */
    uint32_t capacity = 64;
    char **files = malloc(capacity * sizeof(char *));
    uint32_t count = 0;

    struct dirent *entry;
    while ((entry = readdir(dir)) != NULL) {
        const char *name = entry->d_name;
        size_t len = strlen(name);
        if (len > 12 && strcmp(name + len - 12, ".safetensors") == 0) {
            if (count >= capacity) {
                capacity *= 2;
                files = realloc(files, capacity * sizeof(char *));
            }
            size_t dir_len = strlen(dir_path);
            files[count] = malloc(dir_len + 1 + len + 1);
            snprintf(files[count], dir_len + 1 + len + 1,
                     "%s/%s", dir_path, name);
            count++;
        }
    }
    closedir(dir);

    if (count == 0) {
        fprintf(stderr, "no .safetensors files found in %s\n", dir_path);
        free(files);
        return NULL;
    }

    /* Sort alphabetically */
    qsort(files, count, sizeof(char *), compare_strings);

    fprintf(stderr, "Found %u safetensors shards\n", count);

    /* Load each shard */
    sst_sharded_model *model = calloc(1, sizeof(sst_sharded_model));
    if (!model) {
        for (uint32_t i = 0; i < count; i++) free(files[i]);
        free(files);
        return NULL;
    }

    model->n_models = count;
    model->models = calloc(count, sizeof(sst_model *));
    if (!model->models) {
        free(model);
        for (uint32_t i = 0; i < count; i++) free(files[i]);
        free(files);
        return NULL;
    }

    uint64_t total_tensors = 0;
    for (uint32_t i = 0; i < count; i++) {
        fprintf(stderr, "  Loading shard %u/%u: %s\n", i + 1, count, files[i]);
        model->models[i] = sst_model_load(files[i]);
        if (!model->models[i]) {
            fprintf(stderr, "  Failed to load shard %u\n", i);
            sst_sharded_model_free(model);
            for (uint32_t j = 0; j < count; j++) free(files[j]);
            free(files);
            return NULL;
        }
        total_tensors += model->models[i]->n_tensors;
        fprintf(stderr, "    %lu tensors\n", model->models[i]->n_tensors);
    }

    fprintf(stderr, "Total: %lu tensors across %u shards\n", total_tensors, count);

    for (uint32_t i = 0; i < count; i++) free(files[i]);
    free(files);

    return model;
}

void sst_sharded_model_free(sst_sharded_model *m) {
    if (!m) return;
    if (m->models) {
        for (uint64_t i = 0; i < m->n_models; i++) sst_model_free(m->models[i]);
        free(m->models);
    }
    free(m);
}

sst_tensor *sst_sharded_model_find_tensor(sst_sharded_model *m, const char *name) {
    if (!m || !name) return NULL;
    for (uint64_t i = 0; i < m->n_models; i++) {
        sst_tensor *t = sst_model_find_tensor(m->models[i], name);
        if (t) return t;
    }
    return NULL;
}

void *sst_sharded_model_tensor_data(sst_sharded_model *m, sst_tensor *t) {
    if (!m || !t) return NULL;
    /* Find the model that actually contains this tensor by name */
    for (uint64_t i = 0; i < m->n_models; i++) {
        sst_tensor *found = sst_model_find_tensor(m->models[i], t->name);
        if (found == t) {
            return sst_model_tensor_data(m->models[i], t);
        }
    }
    return NULL;
}

/* ========================================================================
 * Format helpers
 * ======================================================================== */

float sst_nvfp4_scale_to_float(uint8_t s) {
    int sign = (s >> 7) & 1;
    int exp = (s >> 3) & 0xF;
    int man = s & 0x7;
    float v;
    if (exp == 0)       v = (man / 8.0f) * powf(2.0f, -6);
    else if (exp == 0xF) v = (man == 0) ? INFINITY : NAN;
    else                v = (1.0f + man / 8.0f) * powf(2.0f, exp - 15);
    return sign ? -v : v;
}

float sst_fp8_e4m3_to_float(uint8_t v) { return sst_nvfp4_scale_to_float(v); }

const char *sst_model_get_metadata_string(sst_model *m, const char *k) { (void)m; (void)k; return NULL; }
int64_t sst_model_get_metadata_int(sst_model *m, const char *k) { (void)m; (void)k; return -1; }
double sst_model_get_metadata_float(sst_model *m, const char *k) { (void)m; (void)k; return -1.0; }
