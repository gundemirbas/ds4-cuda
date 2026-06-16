/**
 * Test safetensors parser - minimal version
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "ds4_safetensors.h"

int main(void) {
    printf("=== Safetensors Test ===\n\n");

    /* Write a minimal safetensors file */
    const char *path = "/tmp/test.sst";
    FILE *fp = fopen(path, "wb");
    if (!fp) { perror("fopen"); return 1; }

    /* Build JSON header by hand - safetensors format: weight0 F32 4 bytes */
    const char *hdr = "{\"w\":{\"dtype\":\"F32\",\"shape\":[1],\"data_offsets\":[0,4]}}";
    uint64_t hdr_len = strlen(hdr) + 1;  /* includes null */
    fwrite(&hdr_len, 8, 1, fp);
    fwrite(hdr, hdr_len, 1, fp);
    float val = 42.0f;
    fwrite(&val, 4, 1, fp);
    fclose(fp);

    /* Load */
    sst_model *m = sst_model_load(path);
    if (!m) { fprintf(stderr, "load failed\n"); return 1; }

    printf("tensors: %lu\n", m->n_tensors);
    printf("header:  %lu\n", m->header_size);
    printf("data:    %lu\n", m->data_size);

    if (m->n_tensors != 1) { fprintf(stderr, "expected 1 tensor\n"); return 1; }

    sst_tensor *t = &m->tensors[0];
    printf("name:    %s\n", t->name);
    printf("dtype:   %d\n", t->dtype);
    printf("size:    %lu\n", t->data_size);

    float *data = sst_model_tensor_data(m, t);
    if (!data) { fprintf(stderr, "data ptr null\n"); return 1; }
    printf("value:   %f\n", *data);

    /* Find by name */
    sst_tensor *found = sst_model_find_tensor(m, "w");
    if (!found) { fprintf(stderr, "find failed\n"); return 1; }
    printf("found:   %s\n", found->name);

    /* Dtype tests */
    printf("\ndtype size F32:   %zu\n", sst_dtype_size(SST_DTYPE_F32));
    printf("dtype size NVFP4: %zu\n", sst_dtype_size(SST_DTYPE_NVFP4));
    printf("dtype bits F32:   %u\n", sst_dtype_bits(SST_DTYPE_F32));
    printf("dtype bits NVFP4: %u\n", sst_dtype_bits(SST_DTYPE_NVFP4));
    printf("elements: %lu\n", sst_tensor_elements(t));

    sst_model_free(m);
    remove(path);

    printf("\n=== PASS ===\n");
    return 0;
}
