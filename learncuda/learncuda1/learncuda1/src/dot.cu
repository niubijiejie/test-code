#include  "dot.h"
#define imin(a,b) (a<b?a:b)
#include "book.h"

const int N = 33*1024;
const int threadsPerBlock = 256;
const int blocksPerGrid = imin(32, (N+threadsPerBlock-1)/threadsPerBlock);
__global__ void dodot(float *a, float *b, float *c);

void dot()
{
    //int a[N],b[N],c[N];
    //CPUBitmap bitmap(16,16);
    float *a,*b,c,*partial_c;
    float *dev_a,*dev_b,*dev_partial_c;

    a=(float*)malloc(N*sizeof(float));
    b=(float*)malloc(N*sizeof(float));
    partial_c = (float*)malloc(blocksPerGrid*sizeof(float));

    HANDLE_ERROR(cudaMalloc((void **)&dev_a, N*sizeof(float)));
    HANDLE_ERROR(cudaMalloc((void **)&dev_b, N*sizeof(float)));
    HANDLE_ERROR(cudaMalloc((void **)&dev_partial_c, N*sizeof(float)));

    for(int i=0;i<N;i++)
    {
        a[i]=i;
        b[i]=i*i;
    }

    HANDLE_ERROR(cudaMemcpy(dev_a, a, N*sizeof(float),cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpy(dev_b, b, N*sizeof(float),cudaMemcpyHostToDevice));
    //HANDLE_ERROR(cudaMemcpy(dev_c, c, N*sizeof(int),cudaMemcpyHostToDevice));
    dodot<<<blocksPerGrid, threadsPerBlock>>>(dev_a,dev_b,dev_partial_c);

    HANDLE_ERROR(cudaMemcpy(partial_c, dev_partial_c, blocksPerGrid*sizeof(float),cudaMemcpyDeviceToHost));
    c=0;
    for(int i=0;i<blocksPerGrid;i++)
    {
        //printf("%d + %d= %d\n", a[i], b[i], c[i]);
        c+=partial_c[i];
    }
    printf("%.6g\n",c);

    cudaFree(dev_a);
    cudaFree(dev_b);
    cudaFree(dev_partial_c);

    free(a);
    free(b);
    free(partial_c);
}
__global__ void dodot(float *a, float *b, float *c)
{
    __shared__ float cache[threadsPerBlock];

    int tid =threadIdx.x + blockIdx.x*blockDim.x;
    int cacheIndex = threadIdx.x;

    float tmp = 0;

    while(tid<N)
    {
        tmp += a[tid]*b[tid];
        tid += blockDim.x*gridDim.x;
    }
    cache[cacheIndex] = tmp;

    __syncthreads();

    int i = blockDim.x/2;
    while(i!=0)
    {
        if(cacheIndex<i)
            cache[cacheIndex] += cache[cacheIndex+i];
        __syncthreads();
        i=i/2;
    }

    if(cacheIndex==0)
        c[blockIdx.x]=cache[0];
}
