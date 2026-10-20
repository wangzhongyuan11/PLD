/* Fixed frozen-model boundary layouts only; no CNN arithmetic runs on PS. */
#include <stdint.h>
#include <stddef.h>
#if defined(__ARM_NEON)
#include <arm_neon.h>
#endif
#define PIXELS (256u * 256u)

int acdc_layout_abi(void) { return 1; }

int acdc_pack_input(uint8_t *dst, size_t capacity,
                    const int8_t *src, size_t length) {
    if (!dst || !src || capacity != 8u*PIXELS || length != 3u*PIXELS)
        return 1;
    unsigned bad=0;
#if defined(__ARM_NEON)
    uint8x16_t zero=vdupq_n_u8(0), invalid=vdupq_n_u8(128), errors=zero;
    for (size_t i=0;i<PIXELS;i+=16) {
        uint8x16_t a=vld1q_u8((const uint8_t *)src+i);
        uint8x16_t b=vld1q_u8((const uint8_t *)src+PIXELS+i);
        uint8x16_t c=vld1q_u8((const uint8_t *)src+2*PIXELS+i);
        errors=vorrq_u8(errors,vorrq_u8(vceqq_u8(a,invalid),
                    vorrq_u8(vceqq_u8(b,invalid),vceqq_u8(c,invalid))));
        uint8x16x2_t ab=vzipq_u8(a,b), cz=vzipq_u8(c,zero);
        for (int half=0;half<2;half++) {
            uint16x8x4_t packed;
            packed.val[0]=vreinterpretq_u16_u8(ab.val[half]);
            packed.val[1]=vreinterpretq_u16_u8(cz.val[half]);
            packed.val[2]=vdupq_n_u16(0);packed.val[3]=vdupq_n_u16(0);
            vst4q_u16((uint16_t *)(dst+8*i+64*half),packed);
        }
    }
    uint8_t flags[16];vst1q_u8(flags,errors);
    for (int i=0;i<16;i++)bad|=flags[i];
#else
    for (size_t i=0;i<PIXELS;i++) {
        for (size_t c=0;c<3;c++) {
            int8_t value=src[c*PIXELS+i];bad|=(value==INT8_MIN);
            dst[8*i+c]=(uint8_t)value;
        }
        for (size_t c=3;c<8;c++)dst[8*i+c]=0;
    }
#endif
    return bad ? 2 : 0;
}

int acdc_unpack_head(int8_t *dst, size_t capacity,
                     const uint8_t *src, size_t length) {
    if (!dst || !src || capacity != 4u*PIXELS || length != 8u*PIXELS)
        return 1;
    unsigned bad=0;
#if defined(__ARM_NEON)
    uint8x16_t invalid=vdupq_n_u8(128), errors=vdupq_n_u8(0);
    uint16x8_t padding=vdupq_n_u16(0);
    for (size_t i=0;i<PIXELS;i+=16) {
        uint16x8x4_t lo=vld4q_u16((const uint16_t *)(src+8*i));
        uint16x8x4_t hi=vld4q_u16((const uint16_t *)(src+8*i+64));
        padding=vorrq_u16(padding,vorrq_u16(vorrq_u16(lo.val[2],lo.val[3]),
                                           vorrq_u16(hi.val[2],hi.val[3])));
        uint8x16x2_t ab=vuzpq_u8(vreinterpretq_u8_u16(lo.val[0]),
                                vreinterpretq_u8_u16(hi.val[0]));
        uint8x16x2_t cd=vuzpq_u8(vreinterpretq_u8_u16(lo.val[1]),
                                vreinterpretq_u8_u16(hi.val[1]));
        for (int c=0;c<4;c++) {
            uint8x16_t v=c<2?ab.val[c]:cd.val[c-2];
            errors=vorrq_u8(errors,vceqq_u8(v,invalid));
            vst1q_u8((uint8_t *)dst+c*PIXELS+i,v);
        }
    }
    uint8_t flags[16];vst1q_u8(flags,vorrq_u8(errors,vreinterpretq_u8_u16(padding)));
    for (int i=0;i<16;i++)bad|=flags[i];
#else
    for (size_t i=0;i<PIXELS;i++) {
        for (size_t c=0;c<4;c++) {
            uint8_t value=src[8*i+c];bad|=(value==128);
            dst[c*PIXELS+i]=(int8_t)value;
        }
        for (size_t c=4;c<8;c++)bad|=src[8*i+c];
    }
#endif
    return bad ? 2 : 0;
}
