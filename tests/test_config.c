/**
 * Test for ds4 model config parser
 *
 * Creates a mock config.json and parses it.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "ds4.h"

int main() {
    printf("=== Model Config Test ===\n\n");

    /* Create a temporary config file */
    const char *config_json =
        "{"
        "  \"n_layers\": 43,"
        "  \"n_heads\": 32,"
        "  \"n_kv_heads\": 1,"
        "  \"head_dim\": 512,"
        "  \"n_experts\": 256,"
        "  \"n_activated_experts\": 6,"
        "  \"vocab_size\": 129280,"
        "  \"hidden_size\": 6144,"
        "  \"intermediate_size\": 12288,"
        "  \"n_mtp_layers\": 1,"
        "  \"max_seq_len\": 65536,"
        "  \"rope_dim\": 512,"
        "  \"rope_freq_base\": 10000,"
        "  \"quantization\": \"NVFP4\""
        "}";

    const char *tmp_path = "/tmp/ds4_config_test.json";
    FILE *fp = fopen(tmp_path, "w");
    if (!fp) { perror("fopen"); return 1; }
    fwrite(config_json, 1, strlen(config_json), fp);
    fclose(fp);

    ds4_model_config *cfg = ds4_config_load(tmp_path);
    if (!cfg) {
        printf("FAIL: ds4_config_load returned NULL\n");
        return 1;
    }

    int ok = 1;
    if (cfg->n_layers != 43)            { printf("FAIL: n_layers=%d (expected 43)\n", cfg->n_layers); ok = 0; }
    if (cfg->n_heads != 32)             { printf("FAIL: n_heads=%d (expected 32)\n", cfg->n_heads); ok = 0; }
    if (cfg->n_kv_heads != 1)           { printf("FAIL: n_kv_heads=%d (expected 1)\n", cfg->n_kv_heads); ok = 0; }
    if (cfg->head_dim != 512)           { printf("FAIL: head_dim=%d (expected 512)\n", cfg->head_dim); ok = 0; }
    if (cfg->n_experts != 256)          { printf("FAIL: n_experts=%d (expected 256)\n", cfg->n_experts); ok = 0; }
    if (cfg->n_activated_experts != 6)  { printf("FAIL: n_activated_experts=%d (expected 6)\n", cfg->n_activated_experts); ok = 0; }
    if (cfg->vocab_size != 129280)      { printf("FAIL: vocab_size=%d (expected 129280)\n", cfg->vocab_size); ok = 0; }
    if (cfg->hidden_size != 6144)       { printf("FAIL: hidden_size=%d (expected 6144)\n", cfg->hidden_size); ok = 0; }
    if (cfg->intermediate_size != 12288){ printf("FAIL: intermediate_size=%d (expected 12288)\n", cfg->intermediate_size); ok = 0; }
    if (cfg->n_mtp_layers != 1)         { printf("FAIL: n_mtp_layers=%d (expected 1)\n", cfg->n_mtp_layers); ok = 0; }
    if (cfg->max_seq_len != 65536)      { printf("FAIL: max_seq_len=%d (expected 65536)\n", cfg->max_seq_len); ok = 0; }
    if (cfg->rope_dim != 512)           { printf("FAIL: rope_dim=%d (expected 512)\n", cfg->rope_dim); ok = 0; }
    if (cfg->rope_freq_base != 10000)   { printf("FAIL: rope_freq_base=%ld (expected 10000)\n", cfg->rope_freq_base); ok = 0; }
    if (!cfg->quantization || strcmp(cfg->quantization, "NVFP4") != 0) {
        printf("FAIL: quantization=%s (expected NVFP4)\n", cfg->quantization ? cfg->quantization : "NULL");
        ok = 0;
    }

    if (ok) {
        printf("All fields match!\n");
    }

    ds4_config_free(cfg);
    remove(tmp_path);

    printf("\n=== %s ===\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}
