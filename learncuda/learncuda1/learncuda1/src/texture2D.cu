#include  "texture2D.h"

#include "book.h"
#include "cpu_anim.h"
#define DIM 1024
#define PI 3.1415926535897932f
#define MAX_TEMP 1.0
#define MIN_TEMP 0.0001f
#define SPEED 0.25f

texture<float,2>  texConstSrc;
texture<float,2>  texIn;
texture<float,2>  texOut;
struct DataBlock
{
    unsigned char *output_bitmap;
    float *dev_inSrc;
    float *dev_outSrc;
    float *dev_constSrc;
    CPUAnimBitmap *bitmap;

    cudaEvent_t start,stop;
    float totalTime;
    float frames;
};

__global__ void kenel(float *ptr);
__global__ void blend_kenel(float *dst,bool dstOut);

void anim_gpu(DataBlock *d, int ticks);
void ainm_exit(DataBlock *d);
void heat()
{
    DataBlock data;
    CPUAnimBitmap bitmap(DIM,DIM,&data);
    data.bitmap = &bitmap;
    data.totalTime = 0;
    data.frames = 0;

    HANDLE_ERROR(cudaEventCreate(&data.start));
    HANDLE_ERROR(cudaEventCreate(&data.stop));

    HANDLE_ERROR(cudaMalloc((void**)&data.dev_inSrc, bitmap.image_size()));
    HANDLE_ERROR(cudaMalloc((void**)&data.output_bitmap, bitmap.image_size()));
    HANDLE_ERROR(cudaMalloc((void**)&data.dev_outSrc, bitmap.image_size()));
    HANDLE_ERROR(cudaMalloc((void**)&data.dev_constSrc, bitmap.image_size()));

    cudaChannelFormatDesc desc = cudaCreateChannelDesc<float>();
    HANDLE_ERROR(cudaBindTexture2D(NULL,texConstSrc,data.dev_constSrc,desc,DIM,DIM,sizeof(float)*DIM));
    HANDLE_ERROR(cudaBindTexture2D(NULL,texIn,data.dev_inSrc,desc,DIM,DIM,sizeof(float)*DIM));
    HANDLE_ERROR(cudaBindTexture2D(NULL,texOut,data.dev_outSrc,desc,DIM,DIM,sizeof(float)*DIM));

    float *temp = (float*)malloc(bitmap.image_size());
    for(int i=0; i<DIM*DIM; i++){
        temp[i] = 0;
        int x=i%DIM;
        int y=i/DIM;
        if((x>300)&&(x<600)&&(y>310)&&(y<601))
            temp[i]=MAX_TEMP;
    }
    temp[DIM*100+100] = (MAX_TEMP+MIN_TEMP)/2;
    temp[DIM*700+100] = MIN_TEMP;
    temp[DIM*300+300] = MIN_TEMP;
    temp[DIM*200+700] = MIN_TEMP;
    for (int y=800; y<900; y++) {
        for (int x=400; x<500; x++) {
            temp[x+y*DIM] = MIN_TEMP;
        }
    }
    HANDLE_ERROR(cudaMemcpy(data.dev_constSrc,temp,bitmap.image_size(),cudaMemcpyHostToDevice));

    for (int y=800; y<DIM; y++) {
        for (int x=0; x<200; x++) {
            temp[x+y*DIM] = MAX_TEMP;
        }
    }
    HANDLE_ERROR(cudaMemcpy(data.dev_inSrc,temp,bitmap.image_size(),cudaMemcpyHostToDevice));

    free(temp);
    bitmap.anim_and_exit((void(*)(void*,int))anim_gpu,(void(*)(void*))ainm_exit);
}
void anim_gpu(DataBlock *d, int ticks)
{
    HANDLE_ERROR(cudaEventRecord(d->start,0));
    dim3 grid(DIM/16,DIM/16);
    dim3 threads(16,16);
    CPUAnimBitmap *bitmap = d->bitmap;

    volatile bool dstOut = true;
    for(int i=0;i<90;i++){
        float *in,*out;
        if(dstOut){
            in = d->dev_inSrc;
            out = d->dev_outSrc;
        }
        else{
            in = d->dev_outSrc;
            out = d->dev_inSrc;
        }
        kenel<<<grid,threads>>>(in);
        blend_kenel<<<grid,threads>>>(out,dstOut);
        dstOut = !dstOut;
    }
    float_to_color<<<grid,threads>>>(d->output_bitmap,d->dev_inSrc);

    HANDLE_ERROR(cudaMemcpy(bitmap->get_ptr(), d->output_bitmap, bitmap->image_size(),cudaMemcpyDeviceToHost));

    HANDLE_ERROR(cudaEventRecord(d->stop,0));
    HANDLE_ERROR(cudaEventSynchronize(d->stop));

    float elapsedTime;
    HANDLE_ERROR(cudaEventElapsedTime(&elapsedTime,d->start,d->stop));
    d->totalTime += elapsedTime;
    ++d->frames;
    printf("Time to generate: %3.1f ms\n", d->totalTime/d->frames);
}
void ainm_exit(DataBlock *d)
{
    cudaUnbindTexture(texIn);
    cudaUnbindTexture(texConstSrc);
    cudaUnbindTexture(texOut);
    cudaFree(d->dev_constSrc);
    cudaFree(d->dev_inSrc);
    cudaFree(d->dev_outSrc);

    HANDLE_ERROR(cudaEventDestroy(d->start));
    HANDLE_ERROR(cudaEventDestroy(d->stop));
}
__global__ void kenel(float *ptr)
{
    int x = threadIdx.x + blockIdx.x*blockDim.x;
    int y = threadIdx.y + blockIdx.y*blockDim.y;
    int offset = x + y*blockDim.x*gridDim.x;

    float c = tex2D(texConstSrc,x,y);
    if(c!=0)
        ptr[offset] = c;
}
__global__ void blend_kenel(float *dst,bool dstOut)
{
    int x = threadIdx.x + blockIdx.x*blockDim.x;
    int y = threadIdx.y + blockIdx.y*blockDim.y;
    int offset = x + y*blockDim.x*gridDim.x;

    float t,l,c,r,b;
    if(dstOut){
        t = tex2D(texIn,x,y-1);
        l = tex2D(texIn,x-1,y);
        c = tex2D(texIn,x,y);
        r = tex2D(texIn,x+1,y);
        b = tex2D(texIn,x,y+1);
    }
    else{
        t = tex2D(texOut,x,y-1);
        l = tex2D(texOut,x-1,y);
        c = tex2D(texOut,x,y);
        r = tex2D(texOut,x+1,y);
        b = tex2D(texOut,x,y+1);
    }
    dst[offset] = c + SPEED * (t+b+r+l - 4*c);
}
