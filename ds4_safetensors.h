/**
 * Safetensors Parser
 *
 * Desteklenen formatlar:
 *   - F32, F16, BF16
 *   - FP8 E4M3
 *   - NVFP4, MXFP4
 */

#ifndef DS4_SAFETENSORS_H
#define DS4_SAFETENSORS_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
    SST_DTYPE_F32,
    SST_DTYPE_F16,
    SST_DTYPE_BF16,
    SST_DTYPE_F8_E4M3,
    SST_DTYPE_NVFP4,
    SST_DTYPE_MXFP4,
    SST_DTYPE_UNKNOWN,
} sst_dtype;

typedef struct {
    char *name;
    sst_dtype dtype;
    uint64_t ndim;
    uint64_t shape[8];
    uint64_t data_offset;
    uint64_t data_size;
} sst_tensor;

typedef struct {
    int fd;
    uint8_t *map;
    uint64_t file_size;
    uint64_t header_size;
    uint64_t data_offset;
    uint64_t data_size;
    uint64_t n_tensors;
    sst_tensor *tensors;
    void *index;
} sst_model;

/* Single file */
sst_model *sst_model_load(const char *path);
void sst_model_free(sst_model *model);
sst_tensor *sst_model_find_tensor(sst_model *model, const char *name);
void *sst_model_tensor_data(sst_model *model, sst_tensor *tensor);

/* Sharded (multi-file) */
typedef struct {
    sst_model **models;
    uint64_t n_models;
    void *global_index;
} sst_sharded_model;

sst_sharded_model *sst_sharded_model_load(const char *dir_path);
void sst_sharded_model_free(sst_sharded_model *model);
sst_tensor *sst_sharded_model_find_tensor(sst_sharded_model *model, const char *name);

/* Tensor Helpers */
uint64_t sst_tensor_elements(const sst_tensor *tensor);
uint64_t sst_tensor_bytes(const sst_tensor *tensor);
void *sst_sharded_model_tensor_data(sst_sharded_model *model, sst_tensor *tensor);

/* Dtype Helpers */
size_t sst_dtype_size(sst_dtype dtype);
uint32_t sst_dtype_block_size(sst_dtype dtype);
uint32_t sst_dtype_bits(sst_dtype dtype);

/* Format Helpers */
float sst_nvfp4_scale_to_float(uint8_t scale);
float sst_fp8_e4m3_to_float(uint8_t value);

#ifdef __cplusplus
}
#endif

#endif /* DS4_SAFETENSORS_H */
