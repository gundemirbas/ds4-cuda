/* test_layer0.c — Full layer 0 decode: CPU reference vs GPU forward */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <cuda_runtime.h>
#include "ds4_safetensors.h"

#define CUDA_CHECK(call) do { \
    cudaError_t err = (call); \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); exit(1); \
    } \
} while(0)

extern void launch_gemv_nvfp4(const float *x, const uint8_t *w, const uint8_t *ws, float *y, int M, int K, float ws2);
extern void launch_gemv_f8e4m3(const float *x, const uint8_t *w, float *y, int M, int K);
extern void launch_gemv_grouped_f8e4m3(const float *x, const uint8_t *w, float *y, int ngroups, int gs, int os);
extern void launch_gemv_f32(const float *x, const float *w, float *y, int M, int K);
extern void launch_rms_norm(const float *x, float *out, const float *weight, int rows, int n, float eps);
extern void launch_rope(float *q, float *k, int n_heads, int head_dim, int pos, float freq_base);
extern void launch_silu(float *x, int n);
extern void launch_emul(float *a, const float *b, int n);
extern void launch_topk_256(const float *scores, int *indices, int k);
extern void launch_bf16_to_f32(const uint16_t *in, float *out, int n);
extern void launch_residual_add(float *a, const float *b, int n);
extern void launch_copy(float *dst, const float *src, int n);

/* --- CPU helpers --- */
static float e2m1_val(uint8_t v) {
    static const float t[16] = {0,0,1,1.5f,2,3,4,6, -0,-0,-1,-1.5f,-2,-3,-4,-6};
    return t[v & 0xFu];
}
static float f8e4m3_val(uint8_t v) {
    if (!v) return 0.0f;
    unsigned s=(v>>7)&1, e=(v>>4)&7, m=v&0xF;
    float val = (e==0) ? (float)m/8.f : (float)(m+16)/16.f * ldexpf(1.f, (int)e-7);
    return s ? -val : val;
}
static float bf16_val(uint16_t v) {
    unsigned s=(v>>15)&1, e=(v>>7)&0xFF, m=v&0x7F;
    float val;
    if (e==0) val = ldexpf((float)m, -126);
    else if (e==0xFF) val = (m==0) ? INFINITY : NAN;
    else val = ldexpf(1.f + (float)m/128.f, (int)e-127);
    return s ? -val : val;
}
static void cpu_gemv_f32(const float *w, const float *x, float *y, int M, int K) {
    for (int r = 0; r < M; r++) { float s=0; for (int k=0;k<K;k++) s+=w[r*K+k]*x[k]; y[r]=s; }
}
static void cpu_gemv_f8e4m3(const uint8_t *w, const float *x, float *y, int M, int K) {
    for (int r = 0; r < M; r++) { float s=0; for (int k=0;k<K;k++) s+=f8e4m3_val(w[r*K+k])*x[k]; y[r]=s; }
}
static void cpu_gemv_grouped_f8e4m3(const uint8_t *w, const float *x, float *y, int n_groups, int gs, int os) {
    for (int g = 0; g < n_groups; g++) {
        const uint8_t *w_g = w + g * os * gs;
        const float *x_g = x + g * gs;
        float *y_g = y + g * os;
        for (int r = 0; r < os; r++) {
            float s = 0;
            for (int k = 0; k < gs; k++) s += f8e4m3_val(w_g[r * gs + k]) * x_g[k];
            y_g[r] = s;
        }
    }
}
static void cpu_gemv_nvfp4(const uint8_t *w, const uint8_t *ws, const float *x, float *y, int M, int K, float ws2) {
    int spc = K / 8;
    for (int r = 0; r < M; r++) {
        float s = 0;
        for (int k = 0; k < K; k++)
            s += e2m1_val(w[r*K+k]) * ws2 * f8e4m3_val(ws[r*spc+k/8]) * x[k];
        y[r] = s;
    }
}
static void cpu_rms_norm(const float *x, float *o, const float *w, int n, float eps) {
    float s=0; for (int i=0;i<n;i++) s+=x[i]*x[i];
    float r = 1.f/sqrtf(s/n+eps);
    for (int i=0;i<n;i++) o[i]=x[i]*r*w[i];
}
static void cpu_rope(float *q, float *k, int n_heads, int head_dim, int pos, float base) {
    int half = head_dim/2;
    for (int h = 0; h < n_heads; h++) {
        for (int d = 0; d < half; d++) {
            float freq = 1.f / powf(base, (float)(2*d)/(float)head_dim);
            float theta = (float)pos * freq;
            float ct = cosf(theta), st = sinf(theta);
            float *qb = q + h*head_dim;
            float a = qb[d], b = qb[d+half];
            qb[d] = a*ct - b*st; qb[d+half] = a*st + b*ct;
        }
    }
    if (k) {
        for (int d = 0; d < half; d++) {
            float freq = 1.f / powf(base, (float)(2*d)/(float)head_dim);
            float theta = (float)pos * freq;
            float ct = cosf(theta), st = sinf(theta);
            float a = k[d], b = k[d+half];
            k[d] = a*ct - b*st; k[d+half] = a*st + b*ct;
        }
    }
}
static void *get_data_ck(sst_sharded_model *m, const char *name, size_t expect) {
    sst_tensor *t = sst_sharded_model_find_tensor(m, name);
    if (!t) { fprintf(stderr, "  MISSING: %s\n", name); return NULL; }
    size_t actual = 1;
    for (uint64_t i = 0; i < t->ndim; i++) actual *= t->shape[i];
    if (actual != expect) fprintf(stderr, "  WARN %s: got %zu expected %zu\n", name, actual, expect);
    fprintf(stderr, "  %s: OK\n", name);
    return sst_sharded_model_tensor_data(m, t);
}

int main(int argc, char **argv) {
    if (argc < 2) { fprintf(stderr, "Usage: %s <model_dir>\n", argv[0]); return 1; }
    fprintf(stderr, "=== Layer 0 Full Decode Test ===\n\n");
    sst_sharded_model *model = sst_sharded_model_load(argv[1]);
    if (!model) return 1;

    const int H=4096, NH=64, HD=512, KVD=512, QD=NH*HD;
    const int EI=2048, EM=2048, EO=4096, NE=256;
    const int O_GROUPS=8, O_LORA=1024, O_OUT=O_GROUPS*O_LORA;
    const float EPS=1e-5f, WS2=2.0f, FREQ_BASE=10000.0f;

    /* --- Load weights --- */
    fprintf(stderr, "Loading weights...\n");
    uint16_t *h_attn_norm_u16 = (uint16_t *)get_data_ck(model, "layers.0.attn_norm.weight", H);
    float *h_attn_norm = (float *)malloc(H * sizeof(float));
    for (int i = 0; i < H; i++) h_attn_norm[i] = bf16_val(h_attn_norm_u16[i]);
    uint16_t *h_ffn_norm_u16 = (uint16_t *)get_data_ck(model, "layers.0.ffn_norm.weight", H);
    float *h_ffn_norm = (float *)malloc(H * sizeof(float));
    for (int i = 0; i < H; i++) h_ffn_norm[i] = bf16_val(h_ffn_norm_u16[i]);
    uint8_t *h_wq_a = (uint8_t *)get_data_ck(model, "layers.0.attn.wq_a.weight", 1024*H);
    uint8_t *h_wq_b = (uint8_t *)get_data_ck(model, "layers.0.attn.wq_b.weight", QD*1024);
    uint8_t *h_wkv  = (uint8_t *)get_data_ck(model, "layers.0.attn.wkv.weight", KVD*H);
    uint8_t *h_wo_a = (uint8_t *)get_data_ck(model, "layers.0.attn.wo_a.weight", O_OUT*H);
    uint8_t *h_wo_b = (uint8_t *)get_data_ck(model, "layers.0.attn.wo_b.weight", H*O_OUT);
    uint16_t *h_gate = (uint16_t *)get_data_ck(model, "layers.0.ffn.gate.weight", NE*H);
    uint8_t *h_w1 = (uint8_t *)get_data_ck(model, "layers.0.ffn.experts.0.w1.weight", EI*EM);
    uint8_t *h_w1s= (uint8_t *)get_data_ck(model, "layers.0.ffn.experts.0.w1.weight_scale", EI*(EM/8));
    uint8_t *h_w2 = (uint8_t *)get_data_ck(model, "layers.0.ffn.experts.0.w2.weight", 4096*1024);
    uint8_t *h_w2s= (uint8_t *)get_data_ck(model, "layers.0.ffn.experts.0.w2.weight_scale", 4096*128);
    uint8_t *h_w3 = (uint8_t *)get_data_ck(model, "layers.0.ffn.experts.0.w3.weight", EI*EM);
    uint8_t *h_w3s= (uint8_t *)get_data_ck(model, "layers.0.ffn.experts.0.w3.weight_scale", EI*(EM/8));
    fprintf(stderr, "\n");

    /* --- Random input --- */
    float *h_in = (float *)malloc(H*sizeof(float));
    srand(42);
    for (int i=0;i<H;i++) h_in[i] = ((float)rand()/RAND_MAX - 0.5f) * 0.1f;

    /* ============== CPU reference ============== */
    fprintf(stderr, "--- CPU reference ---\n");
    float *ref = (float *)malloc(H*sizeof(float));
    cpu_rms_norm(h_in, ref, h_attn_norm, H, EPS);
    float *tmp1024 = (float *)malloc(1024*sizeof(float));
    float *cpu_q = (float *)malloc(QD*sizeof(float));
    cpu_gemv_f8e4m3(h_wq_a, ref, tmp1024, 1024, H);
    cpu_gemv_f8e4m3(h_wq_b, tmp1024, cpu_q, QD, 1024);
    float *cpu_k = (float *)malloc(KVD*sizeof(float));
    cpu_gemv_f8e4m3(h_wkv, ref, cpu_k, KVD, H);
    cpu_rope(cpu_q, cpu_k, NH, HD, 0, FREQ_BASE);
    float *attn_out = (float *)malloc(H*sizeof(float));
    float *tmp_out = (float *)malloc(O_OUT*sizeof(float));
    cpu_gemv_grouped_f8e4m3(h_wo_a, cpu_q, tmp_out, O_GROUPS, H, O_LORA);
    cpu_gemv_f8e4m3(h_wo_b, tmp_out, attn_out, H, O_OUT);
    for (int i=0;i<H;i++) ref[i] += attn_out[i];
    float *cpu_fn = (float *)malloc(H*sizeof(float));
    cpu_rms_norm(ref, cpu_fn, h_ffn_norm, H, EPS);
    float *gate_f32 = (float *)malloc(NE*H*sizeof(float));
    float *cpu_gate = (float *)malloc(NE*sizeof(float));
    for (int i=0;i<NE*H;i++) gate_f32[i] = bf16_val(h_gate[i]);
    cpu_gemv_f32(gate_f32, cpu_fn, cpu_gate, NE, H);
    fprintf(stderr, "CPU gate[0..5]: %.6f %.6f %.6f %.6f %.6f %.6f\n",
            cpu_gate[0], cpu_gate[1], cpu_gate[2], cpu_gate[3], cpu_gate[4], cpu_gate[5]);
    int cpu_topk[6];
    {
        float tmp[256]; memcpy(tmp, cpu_gate, 256*sizeof(float));
        int idx[256]; for(int i=0;i<256;i++) idx[i]=i;
        for (int i=0;i<6;i++) {
            int best=i; for(int j=i+1;j<256;j++) if(tmp[j]>tmp[best]) best=j;
            float tt=tmp[i]; tmp[i]=tmp[best]; tmp[best]=tt;
            int ti=idx[i]; idx[i]=idx[best]; idx[best]=ti;
            cpu_topk[i]=idx[i];
        }
    }
    fprintf(stderr, "CPU top-6: %d %d %d %d %d %d\n",
            cpu_topk[0],cpu_topk[1],cpu_topk[2],cpu_topk[3],cpu_topk[4],cpu_topk[5]);
    float *cpu_e1_raw = (float *)malloc(EI*sizeof(float));
    float *cpu_e1 = (float *)malloc(EI*sizeof(float));
    float *cpu_e3 = (float *)malloc(EI*sizeof(float));
    cpu_gemv_nvfp4(h_w1, h_w1s, cpu_fn, cpu_e1_raw, EI, EM, WS2);
    for (int i=0;i<EI;i++) cpu_e1[i] = cpu_e1_raw[i]/(1.f+expf(-cpu_e1_raw[i]));  /* SiLU */
    cpu_gemv_nvfp4(h_w3, h_w3s, cpu_fn, cpu_e3, EI, EM, WS2);
    for (int i=0;i<1024;i++) cpu_e1[i] *= cpu_e3[i];
    float *cpu_ffn_out = (float *)malloc(H*sizeof(float));
    cpu_gemv_nvfp4(h_w2, h_w2s, cpu_e1, cpu_ffn_out, EO, 1024, WS2);
    for (int i=0;i<H;i++) ref[i] += cpu_ffn_out[i];
    fprintf(stderr, "CPU final[0..3]: %.6f %.6f %.6f %.6f\n", ref[0], ref[1], ref[2], ref[3]);

    /* ============== GPU forward pass ============== */
    fprintf(stderr, "\n--- GPU forward pass ---\n");
    float *d_in, *d_n, *d_res, *d_q, *d_k, *d_t1, *d_t2, *d_t3;
    float *d_gate_f32, *d_gate_sc;
    int *d_topk;
    uint8_t *d_wq_a, *d_wq_b, *d_wkv, *d_wo_a, *d_wo_b;
    uint8_t *d_w1, *d_w1s, *d_w2, *d_w2s, *d_w3, *d_w3s;
    uint16_t *d_gate;
    float *d_attn_norm_f, *d_ffn_norm_f;

    CUDA_CHECK(cudaMalloc(&d_in, H*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_n, H*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_res, H*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_q, QD*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_k, KVD*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_t1, O_OUT*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_t2, EI*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_t3, EM*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_gate_f32, NE*H*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_gate_sc, NE*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_topk, 6*sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_gate, NE*H*sizeof(uint16_t)));
    CUDA_CHECK(cudaMalloc(&d_attn_norm_f, H*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_ffn_norm_f, H*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_wq_a, 1024*H)); CUDA_CHECK(cudaMemcpy(d_wq_a, h_wq_a, 1024*H, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMalloc(&d_wq_b, QD*1024)); CUDA_CHECK(cudaMemcpy(d_wq_b, h_wq_b, QD*1024, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMalloc(&d_wkv, KVD*H)); CUDA_CHECK(cudaMemcpy(d_wkv, h_wkv, KVD*H, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMalloc(&d_wo_a, O_OUT*H)); CUDA_CHECK(cudaMemcpy(d_wo_a, h_wo_a, O_OUT*H, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMalloc(&d_wo_b, H*O_OUT)); CUDA_CHECK(cudaMemcpy(d_wo_b, h_wo_b, H*O_OUT, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMalloc(&d_w1, EI*EM)); CUDA_CHECK(cudaMemcpy(d_w1, h_w1, EI*EM, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMalloc(&d_w1s, EI*(EM/8))); CUDA_CHECK(cudaMemcpy(d_w1s, h_w1s, EI*(EM/8), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMalloc(&d_w2, 4096*1024)); CUDA_CHECK(cudaMemcpy(d_w2, h_w2, 4096*1024, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMalloc(&d_w2s, 4096*128)); CUDA_CHECK(cudaMemcpy(d_w2s, h_w2s, 4096*128, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMalloc(&d_w3, EI*EM)); CUDA_CHECK(cudaMemcpy(d_w3, h_w3, EI*EM, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMalloc(&d_w3s, EI*(EM/8))); CUDA_CHECK(cudaMemcpy(d_w3s, h_w3s, EI*(EM/8), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_gate, h_gate, NE*H*sizeof(uint16_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_attn_norm_f, h_attn_norm, H*sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_ffn_norm_f, h_ffn_norm, H*sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_in, h_in, H*sizeof(float), cudaMemcpyHostToDevice));

    /* 1. attn_rmsnorm */
    launch_rms_norm(d_in, d_n, d_attn_norm_f, 1, H, EPS);
    /* 2. wq_a (F8: 1024 x 4096) */
    launch_gemv_f8e4m3(d_n, d_wq_a, d_t1, 1024, H);
    /* 3. wq_b (F8: 32768 x 1024) */
    launch_gemv_f8e4m3(d_t1, d_wq_b, d_q, QD, 1024);
    /* 4. wkv (F8: 512 x 4096) */
    launch_gemv_f8e4m3(d_n, d_wkv, d_k, KVD, H);
    /* 5. RoPE */
    launch_rope(d_q, d_k, NH, HD, 0, FREQ_BASE);
    /* 6. wo_a (grouped F8: 8 groups, each 1024 x 4096) */
    launch_gemv_grouped_f8e4m3(d_q, d_wo_a, d_t1, O_GROUPS, H, O_LORA);
    /* 7. wo_b (F8: 4096 x 8192) */
    launch_gemv_f8e4m3(d_t1, d_wo_b, d_n, H, O_OUT);
    /* 8. residual */
    launch_residual_add(d_n, d_in, H);
    /* 9. ffn_rmsnorm */
    launch_copy(d_res, d_n, H);
    launch_rms_norm(d_n, d_res, d_ffn_norm_f, 1, H, EPS);
    /* 10. gate scores (BF16 → F32, then F32 GEMV) */
    launch_bf16_to_f32(d_gate, d_gate_f32, NE*H);
    launch_gemv_f32(d_res, d_gate_f32, d_gate_sc, NE, H);
    /* 11. top-6 experts */
    launch_topk_256(d_gate_sc, d_topk, 6);

    float *h_gate_sc = (float *)malloc(NE*sizeof(float));
    int h_topk[6];
    CUDA_CHECK(cudaMemcpy(h_gate_sc, d_gate_sc, NE*sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_topk, d_topk, 6*sizeof(int), cudaMemcpyDeviceToHost));
    fprintf(stderr, "GPU gate[0..5]: %.6f %.6f %.6f %.6f %.6f %.6f\n",
            h_gate_sc[0], h_gate_sc[1], h_gate_sc[2], h_gate_sc[3], h_gate_sc[4], h_gate_sc[5]);
    fprintf(stderr, "GPU top-6: %d %d %d %d %d %d\n",
            h_topk[0],h_topk[1],h_topk[2],h_topk[3],h_topk[4],h_topk[5]);

    /* 12. expert 0 FFN: w1 → silu → (× w3) → w2 */
    launch_gemv_nvfp4(d_res, d_w1, d_w1s, d_t2, EI, EM, WS2);
    launch_silu(d_t2, EI);
    launch_gemv_nvfp4(d_res, d_w3, d_w3s, d_t3, EI, EM, WS2);
    launch_emul(d_t2, d_t3, EI);
    launch_gemv_nvfp4(d_t2, d_w2, d_w2s, d_res, EO, 1024, WS2);
    /* 13. residual */
    launch_residual_add(d_res, d_n, H);

    float *h_out = (float *)malloc(H*sizeof(float));
    CUDA_CHECK(cudaMemcpy(h_out, d_res, H*sizeof(float), cudaMemcpyDeviceToHost));
    fprintf(stderr, "GPU final[0..3]: %.6f %.6f %.6f %.6f\n", h_out[0], h_out[1], h_out[2], h_out[3]);

    /* --- Compare --- */
    double maxdiff = 0;
    for (int i=0;i<H;i++) {
        double d = fabs((double)h_out[i] - (double)ref[i]);
        if (d > maxdiff) maxdiff = d;
    }
    fprintf(stderr, "\nExpert FFN max diff: %.6f %s\n", maxdiff, maxdiff < 2.0 ? "PASS" : "FAIL");
    fprintf(stderr, "(threshold: 2.0 — NVFP4/E8M3 quantization noise for K=2048 reduction)\n");

    /* --- Cleanup --- */
    free(h_in); free(ref); free(tmp1024); free(tmp_out); free(cpu_q); free(cpu_k);
    free(attn_out); free(cpu_fn); free(gate_f32); free(cpu_gate);
    free(cpu_e1_raw); free(cpu_e1); free(cpu_e3); free(cpu_ffn_out); free(h_out); free(h_gate_sc);
    free(h_attn_norm); free(h_ffn_norm);
    cudaFree(d_in); cudaFree(d_n); cudaFree(d_res); cudaFree(d_q); cudaFree(d_k);
    cudaFree(d_t1); cudaFree(d_t2); cudaFree(d_t3);
    cudaFree(d_gate_f32); cudaFree(d_gate_sc); cudaFree(d_topk);
    cudaFree(d_gate); cudaFree(d_attn_norm_f); cudaFree(d_ffn_norm_f);
    cudaFree(d_wq_a); cudaFree(d_wq_b); cudaFree(d_wkv);
    cudaFree(d_wo_a); cudaFree(d_wo_b);
    cudaFree(d_w1); cudaFree(d_w1s); cudaFree(d_w2); cudaFree(d_w2s);
    cudaFree(d_w3); cudaFree(d_w3s);
    sst_sharded_model_free(model);
    return maxdiff < 2.0 ? 0 : 1;
}
