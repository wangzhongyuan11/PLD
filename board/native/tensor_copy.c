/* NCHW INT8 gather with range validation in the same pass. No DMA/MMIO access.
 * The caller owns valid NumPy storage/strides and must not start DMA on error.
 */
#include <stdint.h>
#include <stddef.h>
#ifdef __ARM_NEON
#include <arm_neon.h>
#endif

#ifdef _WIN32
#define EXPORT __declspec(dllexport)
#else
#define EXPORT __attribute__((visibility("default")))
#endif

EXPORT uint32_t tensor_copy_abi(void) { return 1; }

EXPORT int tensor_copy_i8(int8_t *dst, uint64_t capacity, const int8_t *src,
                         const uint64_t shape[4], const int64_t strides[4]) {
    uint64_t count = 1, offset = 0;
    unsigned invalid = 0;
#ifdef __ARM_NEON
    uint8x16_t invalid_vector = vdupq_n_u8(0);
    const int8x16_t forbidden = vdupq_n_s8(INT8_MIN);
#endif
    if (!dst || !src || !shape || !strides) return 2;
    for (unsigned d = 0; d < 4; ++d) {
        if (!shape[d] || shape[d] > SIZE_MAX / count ||
            strides[d] > PTRDIFF_MAX || strides[d] < PTRDIFF_MIN) return 2;
        count *= shape[d];
    }
    if (count > capacity) return 2;
    for (uint64_t n = 0; n < shape[0]; ++n) {
        const int8_t *batch = src + (ptrdiff_t)n * (ptrdiff_t)strides[0];
        for (uint64_t c = 0; c < shape[1]; ++c) {
            const int8_t *channel = batch + (ptrdiff_t)c * (ptrdiff_t)strides[1];
            for (uint64_t y = 0; y < shape[2]; ++y) {
                const int8_t *row = channel + (ptrdiff_t)y * (ptrdiff_t)strides[2];
                if (strides[3] == 1) {
                    uint64_t x = 0;
#ifdef __ARM_NEON
                    for (; x + 16 <= shape[3]; x += 16) {
                        int8x16_t values = vld1q_s8(row+x);
                        vst1q_s8(dst+offset+x, values);
                        invalid_vector = vorrq_u8(invalid_vector, vceqq_s8(values, forbidden));
                    }
#endif
                    for (; x < shape[3]; ++x) {
                        int8_t value = row[x];
                        dst[offset+x] = value;
                        invalid |= value == INT8_MIN;
                    }
                } else {
                    for (uint64_t x = 0; x < shape[3]; ++x) {
                        int8_t value = row[(ptrdiff_t)x * (ptrdiff_t)strides[3]];
                        dst[offset+x] = value;
                        invalid |= value == INT8_MIN;
                    }
                }
                offset += shape[3];
            }
        }
    }
#ifdef __ARM_NEON
    uint8x8_t invalid_half = vorr_u8(vget_low_u8(invalid_vector), vget_high_u8(invalid_vector));
    invalid_half = vpmax_u8(invalid_half, invalid_half);
    invalid_half = vpmax_u8(invalid_half, invalid_half);
    invalid_half = vpmax_u8(invalid_half, invalid_half);
    invalid |= vget_lane_u8(invalid_half, 0);
#endif
    return invalid ? 1 : 0;
}
