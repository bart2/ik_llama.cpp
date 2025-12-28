#pragma once
// Minimal SIMD mappings for AMX compatibility

#include <immintrin.h>
#include <stdint.h>

#if defined(__AMX_INT8__) && defined(__AVX512VNNI__)
// AMX tile configuration structure
struct amx_tile_config {
    uint8_t palette_id;
    uint8_t start_row;
    uint8_t tile_rows;
    uint8_t tile_cols;
    uint16_t reserved;
};

// Tile load/store macros for AMX operations
#define TILE_LOAD(ptr) _tileloadd64(ptr)
#define TILE_STORE(ptr) _tilestored64(ptr)
#define TILE_DPBSSD(t0, t1) _tiledpbssd(t0, t1)
#define TILE_RELEASE() _tile_release()

#endif
