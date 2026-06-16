/* debug_safetensors.c - Minimal test to verify safetensors data loading */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "ds4_safetensors.h"

int main(int argc, char **argv) {
    if (argc < 2) { fprintf(stderr, "Usage: %s <model_dir>\n", argv[0]); return 1; }
    sst_sharded_model *model = sst_sharded_model_load(argv[1]);
    if (!model) return 1;

    // Find w1 tensor
    sst_tensor *t = sst_sharded_model_find_tensor(model, "layers.0.ffn.experts.0.w1.weight");
    if (!t) { fprintf(stderr, "w1 not found\n"); return 1; }
    fprintf(stderr, "w1 data_offset: %lu, data_size: %lu\n", t->data_offset, t->data_size);

    // Find which model contains w1
    for (uint64_t mi = 0; mi < model->n_models; mi++) {
        sst_tensor *t2 = sst_model_find_tensor(model->models[mi], "layers.0.ffn.experts.0.w1.weight");
        if (t2) {
            fprintf(stderr, "w1 found in model %lu, offset: %lu, size: %lu\n", mi, t2->data_offset, t2->data_size);
            // Print first 16 bytes from model's map at that offset
            uint8_t *ptr = (uint8_t *)model->models[mi]->map + t2->data_offset;
            fprintf(stderr, "  Bytes: ");
            for (int j=0;j<16;j++) fprintf(stderr, "%02x ", ptr[j]);
            fprintf(stderr, "\n");
            // Print as float (little-endian)
            float *fp = (float *)ptr;
            fprintf(stderr, "  Floats: ");
            for (int j=0;j<4;j++) fprintf(stderr, "%f ", fp[j]);
            fprintf(stderr, "\n");
        }
    }

    // Also print from sharded model's tensor data function
    void *data = sst_sharded_model_tensor_data(model, t);
    if (data) {
        uint8_t *ptr = (uint8_t *)data;
        fprintf(stderr, "\nFrom sharded_model_tensor_data:\n");
        fprintf(stderr, "  Bytes: ");
        for (int j=0;j<16;j++) fprintf(stderr, "%02x ", ptr[j]);
        fprintf(stderr, "\n");
        float *fp = (float *)ptr;
        fprintf(stderr, "  Floats: ");
        for (int j=0;j<4;j++) fprintf(stderr, "%f ", fp[j]);
        fprintf(stderr, "\n");
    }

    sst_sharded_model_free(model);
    return 0;
}
