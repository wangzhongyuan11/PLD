#include <stdint.h>
#include <math.h>
#include <stdlib.h>
/* Screen-space depth-buffer rasterization; inputs are validated by Python. */
int surface_raster(const float *v,const uint32_t *idx,const uint8_t *color,int n,uint8_t *out,int width,int height){
 float *depth=malloc(width*height*sizeof(float));if(!depth)return -1;
 for(int i=0;i<width*height;i++){depth[i]=-INFINITY;out[i*3]=8;out[i*3+1]=16;out[i*3+2]=25;}
 for(int t=0;t<n;t++){
  const float *a=v+idx[t*3]*3,*b=v+idx[t*3+1]*3,*c=v+idx[t*3+2]*3;
  float area=(b[0]-a[0])*(c[1]-a[1])-(b[1]-a[1])*(c[0]-a[0]);
  if(fabsf(area)<1e-7f)continue;
  int x0=fmaxf(0,floorf(fminf(a[0],fminf(b[0],c[0])))),x1=fminf(width-1,ceilf(fmaxf(a[0],fmaxf(b[0],c[0]))));
  int y0=fmaxf(0,floorf(fminf(a[1],fminf(b[1],c[1])))),y1=fminf(height-1,ceilf(fmaxf(a[1],fmaxf(b[1],c[1]))));
  float inv=1.f/area;
  float ux=(b[1]-c[1])*inv, uy=(c[0]-b[0])*inv;
  float wx=(c[1]-a[1])*inv, wy=(a[0]-c[0])*inv;
  float px=x0+.5f,py=y0+.5f;
  float ur=((b[0]-px)*(c[1]-py)-(b[1]-py)*(c[0]-px))*inv;
  float wr=((c[0]-px)*(a[1]-py)-(c[1]-py)*(a[0]-px))*inv;
  for(int y=y0;y<=y1;y++,ur+=uy,wr+=wy){
   float u=ur,w=wr;
   for(int x=x0;x<=x1;x++,u+=ux,w+=wx){
    float q=1-u-w;if(u<0||w<0||q<0)continue;
    float z=u*a[2]+w*b[2]+q*c[2];int k=y*width+x;
    if(z>depth[k]){depth[k]=z;for(int j=0;j<3;j++)out[k*3+j]=color[t*3+j];}
   }
  }
 }
 free(depth);return 0;
}
