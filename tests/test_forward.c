/* test_forward.c — NVFP4 GEMV correctness test with real model data */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <cuda_runtime.h>
#include "ds4_safetensors.h"

#define CHECK_CUDA(call) do { \
    cudaError_t err = (call); \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); exit(1); \
    } \
} while(0)

static float f8e4m3_to_f32(uint8_t v) {
    if (v == 0) return 0.0f;
    unsigned s = (v >> 7) & 1u, e = (v >> 4) & 7u, m = v & 0xFu;
    float val = (e == 0) ? (float)m / 8.0f : (float)(m + 16) / 16.0f * ldexpf(1.0f, (int)e - 7);
    return s ? -val : val;
}

static float e2m1_to_f32(uint8_t v) {
    static const float t[16] = {0,0,1,1.5f,2,3,4,6, -0,-0,-1,-1.5f,-2,-3,-4,-6};
    return t[v & 0xFu];
}

static void cpu_gemv_nvfp4(const uint8_t *w, const uint8_t *ws,
                            const float *x, float *y, int M, int K, float ws2) {
    int scales_per_row = K / 8;
    for (int r = 0; r < M; r++) {
        float sum = 0.0f;
        for (int k = 0; k < K; k++) {
            uint8_t p = w[r * K + k];
            float sc = f8e4m3_to_f32(ws[r * scales_per_row + k/8]);
            sum += e2m1_to_f32(p & 0xF) * ws2 * sc * x[k];
        }
        y[r] = sum;
    }
}

extern void launch_gemv_nvfp4(const float *x, const uint8_t *w, const uint8_t *ws,
                               float *y, int M, int K, float ws2);

int main(int argc, char **argv) {
    if (argc < 2) { fprintf(stderr, "Usage: %s <model_dir>\n", argv[0]); return 1; }
    fprintf(stderr, "=== NVFP4 GEMV Test (Layer 0, Expert 0, w1) ===\n");
    
    sst_sharded_model *model = sst_sharded_model_load(argv[1]);
    if (!model) return 1;
    fprintf(stderr, "Loaded %lu shards\n", (unsigned long)model->n_models);
    
    sst_tensor *tw = sst_sharded_model_find_tensor(model, "layers.0.ffn.experts.0.w1.weight");
    sst_tensor *ts = sst_sharded_model_find_tensor(model, "layers.0.ffn.experts.0.w1.weight_scale");
    if (!tw || !ts) { fprintf(stderr, "Tensors not found\n"); return 1; }
    
    fprintf(stderr, "w1: [%zu,%zu] scale: [%zu,%zu]\n", tw->shape[0], tw->shape[1], ts->shape[0], ts->shape[1]);
    int M = (int)tw->shape[0], K = (int)tw->shape[1];
    const uint8_t *hw = (const uint8_t *)sst_sharded_model_tensor_data(model, tw);
    const uint8_t *hs = (const uint8_t *)sst_sharded_model_tensor_data(model, ts);
    if (!hw || !hs) { fprintf(stderr, "Tensor data is NULL\n"); return 1; }
    
    fprintf(stderr, "First 16 w1 bytes: ");
    for (int i = 0; i < 16; i++) fprintf(stderr, "%02x ", hw[i]);
    fprintf(stderr, "\nFirst 8 scale bytes: ");
    for (int i = 0; i < 8; i++) fprintf(stderr, "%02x (%.4f) ", hs[i], f8e4m3_to_f32(hs[i]));
    fprintf(stderr, "\n");
    
    float *hx = (float *)malloc(K * sizeof(float));
    float *hy_cpu = (float *)malloc(M * sizeof(float));
    float *hy_gpu = (float *)malloc(M * sizeof(float));
    srand(42);
    for (int i = 0; i < K; i++) hx[i] = (float)rand() / RAND_MAX - 0.5f;
    
    cpu_gemv_nvfp4(hw, hs, hx, hy_cpu, M, K, 2.0f);
    fprintf(stderr, "CPU GEMV [0..7]: ");
    for (int i = 0; i < 8; i++) fprintf(stderr, "%.6f ", hy_cpu[i]);
    fprintf(stderr, "\n");
    
    uint8_t *dw = NULL, *ds_d = NULL;
    float *dx = NULL, *dy = NULL;
    CHECK_CUDA(cudaMalloc((void**)&dw, (size_t)M*K));
    CHECK_CUDA(cudaMemcpy(dw, hw, (size_t)M*K, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMalloc((void**)&ds_d, (size_t)M*(K/8)));
    CHECK_CUDA(cudaMemcpy(ds_d, hs, (size_t)M*(K/8), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMalloc((void**)&dx, (size_t)K*sizeof(float)));
    CHECK_CUDA(cudaMemcpy(dx, hx, (size_t)K*sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMalloc((void**)&dy, (size_t)M*sizeof(float)));
    
    launch_gemv_nvfp4(dx, dw, ds_d, dy, M, K, 2.0f);
    CHECK_CUDA(cudaDeviceSynchronize());
    CHECK_CUDA(cudaMemcpy(hy_gpu, dy, (size_t)M*sizeof(float), cudaMemcpyDeviceToHost));
    
    fprintf(stderr, "GPU GEMV [0..7]: ");
    for (int i = 0; i < 8; i++) fprintf(stderr, "%.6f ", hy_gpu[i]);
    fprintf(stderr, "\n");
    
    double maxdiff = 0.0;
    for (int i = 0; i < M; i++) {
        double d = fabs((double)hy_cpu[i] - (double)hy_gpu[i]);
        if (d > maxdiff) maxdiff = d;
    }
    fprintf(stderr, "Max diff: %.12f %s\n", maxdiff, maxdiff < 1e-4 ? "PASS" : "FAIL");
    
    free(hx); free(hy_cpu); free(hy_gpu);
    cudaFree(dw); cudaFree(ds_d); cudaFree(dx); cudaFree(dy);
    sst_sharded_model_free(model);
    return maxdiff < 1e-4 ? 0 : 1;
}
