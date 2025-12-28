#pragma once
#include "common.h"

#ifdef __cplusplus
extern "C" {
#endif

#if defined(__AMX_INT8__) && defined(__AVX512VNNI__)

// Initialize AMX hardware (call once at startup)
bool ggml_amx_init(void);

// Check if AMX is available at runtime
bool ggml_amx_is_available(void);

// Check if a quantization type is supported by AMX
bool ggml_amx_supports_type(enum ggml_type type);

// Direct matrix multiplication with AMX
// Performs: dst = src1 @ src0.T
void ggml_amx_mul_mat(
    const struct ggml_tensor * src0,  // Quantized weights
    const struct ggml_tensor * src1,  // Float32 input
    struct ggml_tensor * dst,         // Float32 output
    const struct ggml_compute_params * params);

#endif

#ifdef __cplusplus
}
#endif
