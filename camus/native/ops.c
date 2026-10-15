/* Exact CAMUS data movement and float32 reconstruction; no fast-math/FMA. */
#include <stdint.h>
#include <stddef.h>
#include <stdlib.h>
#include <string.h>
#ifdef __ARM_NEON
#include <arm_neon.h>
#endif

static int checked_row(int8_t *dst,const int8_t *src,size_t count) {
    int bad=0;size_t x=0;
#ifdef __ARM_NEON
    uint8x16_t bads=vdupq_n_u8(0),invalid=vdupq_n_u8(128);
    for(;x+16<=count;x+=16) {
        uint8x16_t v=vld1q_u8((const uint8_t *)src+x);
        vst1q_u8((uint8_t *)dst+x,v);bads=vorrq_u8(bads,vceqq_u8(v,invalid));
    }
    uint64x2_t pair=vreinterpretq_u64_u8(bads);
    bad=(vgetq_lane_u64(pair,0)|vgetq_lane_u64(pair,1))!=0;
#endif
    for(;x<count;x++) {int8_t v=src[x];dst[x]=v;bad|=(v==-128);}
    return bad;
}

int camus_copy(int8_t *dst, size_t capacity, const int8_t *src,
               const int64_t shape[4], const int64_t stride[4]) {
    size_t count=1;
    for(int i=0;i<4;i++) { if(shape[i]<1 || (uint64_t)shape[i]>capacity/count) return 2; count*=shape[i]; }
    if(count>capacity) return 2;
    size_t off=0; int bad=0;
    for(int64_t n=0;n<shape[0];n++) for(int64_t c=0;c<shape[1];c++)
    for(int64_t y=0;y<shape[2];y++) {
        const int8_t *row=src+n*stride[0]+c*stride[1]+y*stride[2];
        if(stride[3]==1) {bad|=checked_row(dst+off,row,(size_t)shape[3]);off+=shape[3];}
        else for(int64_t x=0;x<shape[3];x++) { int8_t v=row[x*stride[3]]; dst[off++]=v; bad|=(v==-128); }
    }
    return bad;
}

int camus_gather(int8_t *dst, const int8_t *src, int c, int h, int w,
                  int y0, int x0, int th, int tw) {
    int bad=0;
    int start=x0<0 ? -x0 : 0,end=w-x0<tw ? w-x0 : tw;
    if(start>tw)start=tw;if(end<start)end=start;
    for(int ch=0;ch<c;ch++) for(int y=0;y<th;y++) {
        int sy=y+y0;
        int8_t *row=dst+((size_t)ch*th+y)*tw;
        if(sy<0 || sy>=h) {memset(row,0,(size_t)tw);continue;}
        memset(row,0,(size_t)start);memset(row+end,0,(size_t)(tw-end));
        if(end>start)bad|=checked_row(row+start,src+((size_t)ch*h+sy)*w+x0+start,(size_t)(end-start));
    }
    return bad;
}

void camus_scatter(int8_t *dst, const int8_t *src, int c, int h, int w,
                    int y0, int x0, int th, int tw) {
    for(int ch=0;ch<c;ch++) for(int y=0;y<th;y++)
        memcpy(dst+((size_t)ch*h+y+y0)*w+x0,src+((size_t)ch*th+y)*tw,(size_t)tw);
}

int camus_restore(const int8_t *head, uint8_t *labels, int h, int w,
                   int top, int left, int nh, float scale,
                   const int32_t *xl, const int32_t *xu, const float *xw,
                   const int32_t *yl, const int32_t *yu, const float *yw) {
    float *horizontal=malloc((size_t)nh*w*sizeof(float));
    float *best=malloc((size_t)h*w*sizeof(float));
    if(!horizontal || !best) { free(horizontal); free(best); return 1; }
    for(int ch=0;ch<4;ch++) {
        for(int y=0;y<nh;y++) {
            const int8_t *row=head+(size_t)ch*65536+(top+y)*256+left;
            for(int x=0;x<w;x++) {
                float a=(float)row[xl[x]]*scale, b=(float)row[xu[x]]*scale;
                horizontal[(size_t)y*w+x]=a*(1.0f-xw[x])+b*xw[x];
            }
        }
        for(int y=0;y<h;y++) {
            const float *a=horizontal+(size_t)yl[y]*w,*b=horizontal+(size_t)yu[y]*w;
            for(int x=0;x<w;x++) {
                float v=a[x]*(1.0f-yw[y])+b[x]*yw[y];
                size_t p=(size_t)y*w+x;
                if(ch==0 || v>best[p]) {best[p]=v;labels[p]=(uint8_t)ch;}
            }
        }
    }
    free(horizontal);free(best);return 0;
}

/* 8-connected flood fill, matching scipy.ndimage.label/fill_holes. */
static int flood(const uint8_t *a,int32_t *marks,int32_t *queue,int h,int w,
                 int seed,int tag,int lv) {
    int tail=1;queue[0]=seed;marks[seed]=tag;
    for(int head=0;head<tail;head++) {
        int p=queue[head],y=p/w,x=p%w;
        for(int dy=-1;dy<=1;dy++) for(int dx=-1;dx<=1;dx++) {
            int yy=y+dy,xx=x+dx;
            if(yy<0 || yy>=h || xx<0 || xx>=w)continue;
            int q=yy*w+xx;
            if(!marks[q] && ((a[q]==1)==lv)) {marks[q]=tag;queue[tail++]=q;}
        }
    }
    return tail;
}

int camus_postprocess(const uint8_t *src,uint8_t *out,int h,int w,int32_t stats[10]) {
    size_t count=(size_t)h*w;
    int32_t *marks=calloc(count,sizeof(int32_t)),*queue=malloc(count*sizeof(int32_t));
    if(!marks || !queue) {free(marks);free(queue);return 1;}
    memset(stats,0,10*sizeof(int32_t));memcpy(out,src,count);
    int before=0,components=0,largest=0,largest_id=0;
    for(size_t p=0;p<count;p++) {if(src[p]>3){free(marks);free(queue);return 2;}before+=(src[p]==1);}
    for(int p=0;p<(int)count;p++) if(out[p]==1 && !marks[p]) {
        int size=flood(out,marks,queue,h,w,p,++components,1);
        if(size>largest) {largest=size;largest_id=components;}
    }
    int secondary=before-largest,island=components>1 && (double)secondary/before<=0.05,changed=0;
    if(island)for(int p=0;p<(int)count;p++) if(out[p]==1 && marks[p]!=largest_id) {out[p]=0;changed++;}
    int retained=before-changed,closed=0,filled=0,hole=0;
    if(retained) {
        memset(marks,0,count*sizeof(int32_t));
        for(int y=0;y<h;y++) for(int edge=0;edge<2;edge++) {
            int p=y*w+(edge ? w-1 : 0);
            if(out[p]!=1 && !marks[p])flood(out,marks,queue,h,w,p,2,0);
        }
        for(int x=0;x<w;x++) for(int edge=0;edge<2;edge++) {
            int p=(edge ? h-1 : 0)*w+x;
            if(out[p]!=1 && !marks[p])flood(out,marks,queue,h,w,p,2,0);
        }
        for(int p=0;p<(int)count;p++) if(out[p]!=1 && !marks[p]) {
            int size=flood(out,marks,queue,h,w,p,1,0),foreign=0;
            for(int j=0;j<size;j++)foreign|=(out[queue[j]]!=0);
            for(int j=0;j<size;j++)marks[queue[j]]=foreign ? 4 : 3;
            if(!foreign)closed+=size;
        }
        hole=closed>0 && (double)closed/retained<=0.01;
        if(hole)for(int p=0;p<(int)count;p++)if(marks[p]==3){out[p]=1;filled++;}
    }
    stats[0]=before;stats[1]=retained+filled;stats[2]=components;stats[3]=largest;
    stats[4]=secondary;stats[5]=island;stats[6]=filled;stats[7]=hole;stats[8]=changed+filled;stats[9]=closed;
    free(marks);free(queue);return 0;
}
