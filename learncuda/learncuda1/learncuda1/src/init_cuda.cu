#include  "init_cuda.h"
#define N 10
#include "book.h"

__global__ void add(int *a, int *b, int *c);

void InitCUDA()
{
    int a[N],b[N],c[N];
    //CPUBitmap bitmap(16,16);
    int *dev_a,*dev_b,*dev_c;
    HANDLE_ERROR(cudaMalloc((void **)&dev_a, N*sizeof(int)));
    HANDLE_ERROR(cudaMalloc((void **)&dev_b, N*sizeof(int)));
    HANDLE_ERROR(cudaMalloc((void **)&dev_c, N*sizeof(int)));
    for(int i=0;i<N;i++)
    {
        a[i]=-i;
        b[i]=i*i;
    }
    HANDLE_ERROR(cudaMemcpy(dev_a, a, N*sizeof(int),cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpy(dev_b, b, N*sizeof(int),cudaMemcpyHostToDevice));
    //HANDLE_ERROR(cudaMemcpy(dev_c, c, N*sizeof(int),cudaMemcpyHostToDevice));
    add<<<N, 1>>>(dev_a,dev_b,dev_c);

    HANDLE_ERROR(cudaMemcpy(c, dev_c, N*sizeof(int),cudaMemcpyDeviceToHost));
    for(int i=0;i<N;i++)
    {
        printf("%d + %d= %d\n", a[i], b[i], c[i]);
    }

    cudaFree(dev_a);
    cudaFree(dev_b);
    cudaFree(dev_c);
}
__global__ void add(int *a, int *b, int *c)
{
    int tid =blockIdx.x;
    if(tid<N)
        c[tid] = a[tid]+b[tid];

}
