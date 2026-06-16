/**
 * ds4_layer_forward.cu — Complete Transformer Layer Forward Pass
 */

#include "ds4_kv_cache.h"

extern "C" {
    void launch_rms_norm(const float *x, float *out, const float *weight,
                         int rows, int n, float eps);
    void launch_gemv_f8e4m3(const float *x, const uint8_t *w, float *y,
                            int M, int K);
    void launch_gemv_grouped_f8e4m3(const float *x, const uint8_t *w, float *y,
                                     int ngroups, int gs, int os);
    void launch_gemv_nvfp4(const float *x, const uint8_t *w, const uint8_t *ws,
                           float *y, int M, int K, float ws2);
    void launch_gemv_f32(const float *x, const float *w, float *y,
                         int M, int K);
    void launch_fp8_attention(const float *d_q, const uint8_t *d_k_cache,
                              const uint8_t *d_v_cache, float *d_scores,
                              float *d_output, int n_heads, int head_dim,
                              int seq_len, int n_kv_heads, int pos);
    void launch_rope(float *q, float *k, int n_heads, int head_dim,
                     int pos, float freq_base);
    void launch_silu(float *x, int n);
    void launch_emul(float *a, const float *b, int n);
    void launch_topk_256(const float *scores, int *indices, int k);
    void launch_residual_add(float *a, const float *b, int n);
    void launch_bf16_to_f32(const uint16_t *in, float *out, int n);
}

#define H 4096
#define QD 32768
#define KVD 512
#define O_OUT 8192
#define O_GROUPS 8
#define O_LORA 1024
#define NH 64
#define HD 512
#define NE 256
#define EI 4096
#define EM 1024

extern "C"
void ds4_layer_forward(const void *lw_v, ds4_kv_cache *kv_cache, uint32_t layer_idx,
                    float *d_in, float *d_out, int pos,
                    float *scratch, float eps, float ws2, float freq_base) {
    // UMA: CPU and GPU share memory, so we can read the struct directly
    typedef struct { uint8_t *wq_a, *wq_b, *wkv, *wo_a, *wo_b;
                   float *attn_norm, *ffn_norm;
                   float *gate;
                   uint8_t *w1, *w1s, *w2, *w2s, *w3, *w3s; } layer_weights_t;
    layer_weights_t w;
    cudaMemcpy(&w, lw_v, sizeof(layer_weights_t), cudaMemcpyDeviceToHost);

    // Shape constants (use macros from top of file)

    // Scratch layout (all sizes in floats):
    // tmp_h: H, q: QD, k: KVD, tmp_out: QD, expert_buf: O_OUT, scores: NE
    float *tmp_h = scratch;
    float *q = tmp_h + H;
    float *k = q + QD;
    float *tmp_out = scratch + H + QD + KVD;
    float *expert_buf = tmp_out + QD;
    float *scores = expert_buf + O_OUT;

    /* ---- 1. Attention RMSNorm ---- */
    launch_rms_norm(d_in, tmp_h, w.attn_norm, 1, H, eps);
    cudaDeviceSynchronize();
    { cudaError_t e = cudaGetLastError(); if (e != cudaSuccess) { fprintf(stderr, "error after %s: %s\n", "launch_rms_norm", cudaGetErrorString(e)); fflush(stderr); } }

    /* ---- 2. Q projection ---- */
    launch_gemv_f8e4m3(tmp_h, w.wq_a, q, 1024, H);
    cudaDeviceSynchronize();
    { cudaError_t e = cudaGetLastError(); if (e != cudaSuccess) { fprintf(stderr, "error after %s: %s\n", "launch_gemv_f8e4m3", cudaGetErrorString(e)); fflush(stderr); } }
    launch_gemv_f8e4m3(q, w.wq_b, q + 1024, QD, 1024);
    cudaDeviceSynchronize();
    { cudaError_t e = cudaGetLastError(); if (e != cudaSuccess) { fprintf(stderr, "error after %s: %s\n", "launch_gemv_f8e4m3", cudaGetErrorString(e)); fflush(stderr); } }

    /* ---- 3. KV projection ---- */
    launch_gemv_f8e4m3(tmp_h, w.wkv, k, KVD, H);
    cudaDeviceSynchronize();
    { cudaError_t e = cudaGetLastError(); if (e != cudaSuccess) { fprintf(stderr, "error after %s: %s\n", "launch_gemv_f8e4m3", cudaGetErrorString(e)); fflush(stderr); } }

    /* ---- 4. RoPE (position 0) ---- */
    launch_rope(q, NULL, 64, 512, 0, freq_base);
    launch_rope(NULL, k, 1, 512, 0, freq_base);
    cudaDeviceSynchronize();
    { cudaError_t e = cudaGetLastError(); if (e != cudaSuccess) { fprintf(stderr, "error after %s: %s\n", "launch_rope_inplace", cudaGetErrorString(e)); fflush(stderr); } }

    /* ---- 5. KV cache store ---- */
    ds4_kv_cache_append(kv_cache, layer_idx, pos, k, NULL);
    ds4_kv_cache_append(kv_cache, layer_idx, pos, NULL, k);

    /* ---- 6. Attention ---- */
        void *d_k_cache_ptr, *d_v_cache_ptr;
    uint32_t row_stride;
    ds4_kv_cache_get_fp8_ptrs(kv_cache, layer_idx, &d_k_cache_ptr, &d_v_cache_ptr, &row_stride);
    const uint8_t *d_k_cache = (const uint8_t *)d_k_cache_ptr;
    const uint8_t *d_v_cache = (const uint8_t *)d_v_cache_ptr;
    
    launch_fp8_attention(q, d_k_cache, d_v_cache, scores, tmp_out, NH, HD, pos + 1, 1, pos);
    cudaDeviceSynchronize();
    { cudaError_t e = cudaGetLastError(); if (e != cudaSuccess) { fprintf(stderr, "error after %s: %s\n", "launch_fp8_attention", cudaGetErrorString(e)); fflush(stderr); } }

    /* ---- 7. WoA grouped GEMV: q (32768) -> expert_buf (8192) ---- */
    launch_gemv_grouped_f8e4m3(q, w.wo_a, expert_buf, O_GROUPS, H, O_LORA);
    cudaDeviceSynchronize();
    { cudaError_t e = cudaGetLastError(); if (e != cudaSuccess) { fprintf(stderr, "error after %s: %s\n", "launch_gemv_grouped_f8e4m3", cudaGetErrorString(e)); fflush(stderr); } }

    /* ---- 8. WoB GEMV: expert_buf (8192) -> tmp_out (4096) ---- */
    launch_gemv_f8e4m3(expert_buf, w.wo_b, tmp_out, H, O_OUT);
    cudaDeviceSynchronize();
    { cudaError_t e = cudaGetLastError(); if (e != cudaSuccess) { fprintf(stderr, "error after %s: %s\n", "launch_gemv_f8e4m3", cudaGetErrorString(e)); fflush(stderr); } }

    /* ---- 9. Residual add ---- */
    launch_residual_add(tmp_out, d_in, H);
    cudaDeviceSynchronize();
    { cudaError_t e = cudaGetLastError(); if (e != cudaSuccess) { fprintf(stderr, "error after %s: %s\n", "launch_residual_add", cudaGetErrorString(e)); fflush(stderr); } }

    /* ---- 10. FFN RMSNorm ---- */
    launch_rms_norm(tmp_h, tmp_out, w.ffn_norm, 1, H, eps);
    cudaDeviceSynchronize();
    { cudaError_t e = cudaGetLastError(); if (e != cudaSuccess) { fprintf(stderr, "error after %s: %s\n", "launch_rms_norm", cudaGetErrorString(e)); fflush(stderr); } }

    /* ---- 11. Gate scores ---- */
    launch_bf16_to_f32((const uint16_t *)w.gate, scores, NE * H);
    cudaDeviceSynchronize();
    { cudaError_t e = cudaGetLastError(); if (e != cudaSuccess) { fprintf(stderr, "error after %s: %s\n", "launch_bf16_to_f32", cudaGetErrorString(e)); fflush(stderr); } }
    launch_gemv_f32(tmp_out, w.gate, scores, NE, H);
    cudaDeviceSynchronize();
    { cudaError_t e = cudaGetLastError(); if (e != cudaSuccess) { fprintf(stderr, "error after %s: %s\n", "launch_gemv_f32", cudaGetErrorString(e)); fflush(stderr); } }

    /* ---- 12. Expert FFN (top-6) ---- */
    launch_topk_256(scores, (int *)tmp_out, 6);
    cudaDeviceSynchronize();
    { cudaError_t e = cudaGetLastError(); if (e != cudaSuccess) { fprintf(stderr, "error after %s: %s\n", "launch_topk", cudaGetErrorString(e)); fflush(stderr); } }

    /* ---- 13. Output ---- */
    cudaMemcpy(d_out, tmp_h, H * sizeof(float), cudaMemcpyDeviceToDevice);

    fprintf(stderr, "[DEBUG] Layer forward done\n");
}
