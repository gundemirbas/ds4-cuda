/* ds4_model_internal.h
 *
 * Internal model structs shared between ds4.c (engine) and
 * ds4_model_safetensors.c (safetensors loader).
 *
 * These are POD types that match what ds4.c's GGUF loader produces.
 * ds4.h exposes the public engine API; these structs are engine-internal.
 */

#ifndef DS4_MODEL_INTERNAL_H
#define DS4_MODEL_INTERNAL_H

#include <stdint.h>
#include <stddef.h>

#define DS4_MAX_DIMS 8

/* String slice (same layout as ds4_str in ds4.h) */
typedef struct {
    const char *ptr;
    size_t len;
} ds4_str;

/* GGUF value types */
enum {
    GGUF_VALUE_UINT8   = 0,
    GGUF_VALUE_INT8    = 1,
    GGUF_VALUE_UINT16  = 2,
    GGUF_VALUE_INT16   = 3,
    GGUF_VALUE_UINT32  = 4,
    GGUF_VALUE_INT32   = 5,
    GGUF_VALUE_UINT64  = 6,
    GGUF_VALUE_INT64   = 7,
    GGUF_VALUE_FLOAT32 = 8,
    GGUF_VALUE_FLOAT64 = 9,
    GGUF_VALUE_BOOL    = 10,
    GGUF_VALUE_STRING  = 11,
    GGUF_VALUE_ARRAY   = 12,
};

/* GGUF tensor types (subset used by ds4.c) */
enum {
    DS4_TENSOR_F32      = 0,
    DS4_TENSOR_F16      = 1,
    DS4_TENSOR_BF16     = 2,
    DS4_TENSOR_F8_E4M3  = 3,
    DS4_TENSOR_NVFP4    = 4,
    DS4_TENSOR_MXFP4    = 5,
    DS4_TENSOR_Q8_0     = 8,
    DS4_TENSOR_IQ2_XXS  = 16,
    DS4_TENSOR_I32      = 26,
};

/* Metadata key-value entry */
typedef struct {
    ds4_str key;
    uint32_t type;
    uint64_t value_pos;
} ds4_kv;

/* Tensor directory entry */
typedef struct {
    ds4_str name;
    uint32_t ndim;
    uint64_t dim[DS4_MAX_DIMS];
    uint32_t type;
    uint64_t rel_offset;
    uint64_t abs_offset;
    uint64_t elements;
    uint64_t bytes;
} ds4_tensor;

/* Model handle (mmap'd file + metadata + tensor directory) */
typedef struct {
    int fd;
    const uint8_t *map;
    uint64_t size;

    uint32_t version;
    uint64_t n_kv;
    uint64_t n_tensors;
    uint64_t alignment;
    uint64_t tensor_data_pos;
    uint64_t max_tensor_bytes;

    ds4_kv *kv;
    ds4_tensor *tensors;

    /* Extra field for safetensors loader — vocab/model data not part of GGUF */
    void *vocab_data;
    char *key_buf;
    char *model_dir;  /* Path to model directory (safetensors) */
} ds4_model;

#endif /* DS4_MODEL_INTERNAL_H */
