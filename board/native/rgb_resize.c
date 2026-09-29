/* RGB-only separable triangle resize candidate, using normalized Q22 weights
 * and an 8-bit horizontal intermediate to match the Pillow numerical contract.
 * Reference: Pillow 9.0.1 src/libImaging/Resample.c (coefficient/rounding rules).
 * This implementation operates on public packed RGB buffers, never PIL internals.
 */
#include <stdint.h>
#include <stddef.h>
#include <math.h>
#ifdef __ARM_NEON
#include <arm_neon.h>
#endif
#ifdef _WIN32
#define EXPORT __declspec(dllexport)
#else
#define EXPORT __attribute__((visibility("default")))
#endif

EXPORT uint32_t rgb_resize_abi(void) { return 1; }

EXPORT int rgb_coefficients(uint32_t source, uint32_t target, uint32_t stride,
                            uint32_t *bounds, int32_t *weights) {
    if (!source || !target || source>16384 || target>4096 || !bounds || !weights) return 1;
    double ratio=(double)source/target;
    double radius=ratio>1.0 ? ratio : 1.0;
    if (stride<(uint32_t)ceil(radius)*2+1) return 1;
    for (uint32_t x=0;x<target;++x) {
        double center=(x+0.5)*ratio, inverse=1.0/radius, total=0;
        int first=(int)(center-radius+0.5), stop=(int)(center+radius+0.5);
        if (first<0) first=0;
        if (stop>(int)source) stop=(int)source;
        bounds[2*x]=(uint32_t)first; bounds[2*x+1]=(uint32_t)(stop-first);
        for (int i=first;i<stop;++i) {
            double distance=fabs((i-center+0.5)*inverse);
            total+=distance<1.0 ? 1.0-distance : 0.0;
        }
        if (total<=0) return 2;
        for (uint32_t i=0;i<stride;++i) {
            double distance=fabs(((double)i+first-center+0.5)*inverse);
            double weight=i<(uint32_t)(stop-first) && distance<1.0 ? 1.0-distance : 0.0;
            weights[(size_t)x*stride+i]=(int32_t)(0.5+(weight/total)*4194304.0);
        }
    }
    return 0;
}

static uint8_t output_byte(uint32_t sum) {
    sum>>=22;
    return (uint8_t)(sum>255 ? 255 : sum);
}

#ifdef __ARM_NEON
static uint32_t sum4(int32x4_t values) {
    int32x2_t halves=vadd_s32(vget_low_s32(values),vget_high_s32(values));
    return (uint32_t)vget_lane_s32(vpadd_s32(halves,halves),0);
}
static int32x4_t widen4(uint16x4_t values) {
    return vreinterpretq_s32_u32(vmovl_u16(values));
}
#endif

EXPORT int rgb_horizontal(const uint8_t *src, uint64_t src_bytes, uint8_t *dst, uint64_t dst_bytes,
                          uint32_t width, uint32_t height, uint32_t target,
                          const uint32_t *bounds, const int32_t *weights, uint32_t stride) {
    if (!src || !dst || !bounds || !weights || !width || !height || !target ||
        width>16384 || height>16384 || target>4096 ||
        (uint64_t)width*height*3>src_bytes || (uint64_t)target*height*3>dst_bytes) return 1;
    for (uint32_t x=0;x<target;++x)
        if (bounds[2*x]>=width || bounds[2*x+1]>stride || bounds[2*x+1]>width-bounds[2*x]) return 1;
    for (uint32_t y=0;y<height;++y) for (uint32_t x=0;x<target;++x) {
        const uint8_t *pixels=src+((size_t)y*width+bounds[2*x])*3;
        const int32_t *coeff=weights+(size_t)x*stride;
        uint32_t count=bounds[2*x+1], i=0;
        uint32_t sums[3]={2097152,2097152,2097152};
#ifdef __ARM_NEON
        int32x4_t accum[3]={vdupq_n_s32(0),vdupq_n_s32(0),vdupq_n_s32(0)};
        for (;i+8<=count;i+=8) {
            uint8x8x3_t channels=vld3_u8(pixels+3*i);
            int32x4_t lo=vld1q_s32(coeff+i),hi=vld1q_s32(coeff+i+4);
            for (unsigned c=0;c<3;++c) {
                uint16x8_t values=vmovl_u8(channels.val[c]);
                accum[c]=vmlaq_s32(accum[c],widen4(vget_low_u16(values)),lo);
                accum[c]=vmlaq_s32(accum[c],widen4(vget_high_u16(values)),hi);
            }
        }
        for (unsigned c=0;c<3;++c) sums[c]+=sum4(accum[c]);
#endif
        for (;i<count;++i) for (unsigned c=0;c<3;++c) sums[c]+=(uint32_t)pixels[3*i+c]*coeff[i];
        for (unsigned c=0;c<3;++c) dst[((size_t)y*target+x)*3+c]=output_byte(sums[c]);
    }
    return 0;
}

EXPORT int rgb_vertical(const uint8_t *src, uint64_t src_bytes, uint8_t *dst, uint64_t dst_bytes,
                        uint32_t width, uint32_t height, uint32_t target,
                        const uint32_t *bounds, const int32_t *weights, uint32_t stride) {
    if (!src || !dst || !bounds || !weights || !width || !height || !target ||
        width>4096 || height>16384 || target>4096 ||
        (uint64_t)width*height*3>src_bytes || (uint64_t)width*target*3>dst_bytes) return 1;
    for (uint32_t y=0;y<target;++y)
        if (bounds[2*y]>=height || bounds[2*y+1]>stride || bounds[2*y+1]>height-bounds[2*y]) return 1;
    uint32_t row_bytes=width*3;
    for (uint32_t y=0;y<target;++y) {
        uint32_t x=0,count=bounds[2*y+1];
        const int32_t *coeff=weights+(size_t)y*stride;
        const uint8_t *first=src+(size_t)bounds[2*y]*row_bytes;
#ifdef __ARM_NEON
        for (;x+16<=row_bytes;x+=16) {
            int32x4_t acc[4]={vdupq_n_s32(2097152),vdupq_n_s32(2097152),
                             vdupq_n_s32(2097152),vdupq_n_s32(2097152)};
            for (uint32_t i=0;i<count;++i) {
                uint8x16_t bytes=vld1q_u8(first+(size_t)i*row_bytes+x);
                uint16x8_t lo=vmovl_u8(vget_low_u8(bytes)),hi=vmovl_u8(vget_high_u8(bytes));
                int32x4_t weight=vdupq_n_s32(coeff[i]);
                acc[0]=vmlaq_s32(acc[0],widen4(vget_low_u16(lo)),weight);
                acc[1]=vmlaq_s32(acc[1],widen4(vget_high_u16(lo)),weight);
                acc[2]=vmlaq_s32(acc[2],widen4(vget_low_u16(hi)),weight);
                acc[3]=vmlaq_s32(acc[3],widen4(vget_high_u16(hi)),weight);
            }
            uint16x8_t lo=vcombine_u16(vqmovun_s32(vshrq_n_s32(acc[0],22)),vqmovun_s32(vshrq_n_s32(acc[1],22)));
            uint16x8_t hi=vcombine_u16(vqmovun_s32(vshrq_n_s32(acc[2],22)),vqmovun_s32(vshrq_n_s32(acc[3],22)));
            vst1q_u8(dst+(size_t)y*row_bytes+x,vcombine_u8(vqmovn_u16(lo),vqmovn_u16(hi)));
        }
#endif
        for (;x<row_bytes;++x) {
            uint32_t sum=2097152;
            for (uint32_t i=0;i<count;++i) sum+=first[(size_t)i*row_bytes+x]*(uint32_t)coeff[i];
            dst[(size_t)y*row_bytes+x]=output_byte(sum);
        }
    }
    return 0;
}
/* Pinned Pillow RGB rows: four bytes/pixel, with the fourth byte ignored.
 * Caller holds the source Image throughout this read-only call. Separate row
 * spans can run concurrently; each writes only its own destination rows. */
EXPORT int rgb_horizontal_rows(const uint8_t *const *rows, uint8_t *dst, uint64_t capacity,
        uint32_t width, uint32_t height, uint32_t target, uint32_t begin, uint32_t end,
        const uint32_t *bounds, const int32_t *weights, uint32_t stride) {
    if (!rows || !dst || !bounds || !weights || !width || !height || !target ||
        width>16384 || height>16384 || target>4096 || begin>=end || end>height ||
        (uint64_t)target*height*3>capacity) return 1;
    for (uint32_t x=0;x<target;++x)
        if (bounds[2*x]>=width || bounds[2*x+1]>stride || bounds[2*x+1]>width-bounds[2*x]) return 1;
    for (uint32_t y=begin;y<end;++y) if (!rows[y]) return 1;
    for (uint32_t y=begin;y<end;++y) for (uint32_t x=0;x<target;++x) {
        const uint8_t *pixels=rows[y]+(size_t)bounds[2*x]*4;
        const int32_t *coeff=weights+(size_t)x*stride;
        uint32_t count=bounds[2*x+1],i=0,sums[3]={2097152,2097152,2097152};
#ifdef __ARM_NEON
        int32x4_t accum[3]={vdupq_n_s32(0),vdupq_n_s32(0),vdupq_n_s32(0)};
        for (;i+8<=count;i+=8) {
            uint8x8x4_t channels=vld4_u8(pixels+4*i);
            int32x4_t lo=vld1q_s32(coeff+i),hi=vld1q_s32(coeff+i+4);
            for (unsigned c=0;c<3;++c) {
                uint16x8_t values=vmovl_u8(channels.val[c]);
                accum[c]=vmlaq_s32(accum[c],widen4(vget_low_u16(values)),lo);
                accum[c]=vmlaq_s32(accum[c],widen4(vget_high_u16(values)),hi);
            }
        }
        for (unsigned c=0;c<3;++c) sums[c]+=sum4(accum[c]);
#endif
        for (;i<count;++i) for (unsigned c=0;c<3;++c) sums[c]+=(uint32_t)pixels[4*i+c]*coeff[i];
        for (unsigned c=0;c<3;++c) dst[((size_t)y*target+x)*3+c]=output_byte(sums[c]);
    }
    return 0;
}
