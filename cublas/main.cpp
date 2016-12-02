
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include <cublas_v2.h>



#define cudacall(call)                                                                                                          \
    do                                                                                                                          \
    {                                                                                                                           \
        cudaError_t err = (call);                                                                                               \
        if(cudaSuccess != err)                                                                                                  \
        {                                                                                                                       \
            fprintf(stderr,"CUDA Error:\nFile = %s\nLine = %d\nReason = %s\n", __FILE__, __LINE__, cudaGetErrorString(err));    \
            cudaDeviceReset();                                                                                                  \
            exit(EXIT_FAILURE);                                                                                                 \
        }                                                                                                                       \
    }                                                                                                                           \
    while (0)

#define cublascall(call)                                                                                        \
    do                                                                                                          \
    {                                                                                                           \
        cublasStatus_t status = (call);                                                                         \
        if(CUBLAS_STATUS_SUCCESS != status)                                                                     \
        {                                                                                                       \
            fprintf(stderr,"CUBLAS Error:\nFile = %s\nLine = %d\nCode = %d\n", __FILE__, __LINE__, status);     \
            cudaDeviceReset();                                                                                  \
            exit(EXIT_FAILURE);                                                                                 \
        }                                                                                                       \
                                                                                                                \
    }                                                                                                           \
    while(0)
void invert(float* src, float* dst, int n)
{
    float* src_d, *dst_d;

    cudacall(cudaMalloc<float>(&src_d,n * n * sizeof(float)));
    cudacall(cudaMemcpy(src_d,src,n * n * sizeof(float),cudaMemcpyHostToDevice));
    cudacall(cudaMalloc<float>(&dst_d,n * n * sizeof(float)));

    cublasHandle_t handle;
    cublascall(cublasCreate_v2(&handle));

    int batchSize = 1;

    int *P, *INFO;

    cudacall(cudaMalloc<int>(&P,n * batchSize * sizeof(int)));
    cudacall(cudaMalloc<int>(&INFO,batchSize * sizeof(int)));

    int lda = n;

    float *A[] = { src_d };
    float** A_d;
    cudacall(cudaMalloc<float*>(&A_d,sizeof(A)));
    cudacall(cudaMemcpy(A_d,A,sizeof(A),cudaMemcpyHostToDevice));
    cublascall(cublasSgetrfBatched(handle,n,A_d,lda,P,INFO,batchSize));

    int INFOh = 0;
    cudacall(cudaMemcpy(&INFOh,INFO,sizeof(int),cudaMemcpyDeviceToHost));

    if(INFOh == n)
    {
        fprintf(stderr, "Factorization Failed: Matrix is singular\n");
        cudaDeviceReset();
        exit(EXIT_FAILURE);
    }

    float* C[] = { dst_d };
    float** C_d;
    cudacall(cudaMalloc<float*>(&C_d,sizeof(C)));
    cudacall(cudaMemcpy(C_d,C,sizeof(C),cudaMemcpyHostToDevice));

    cublascall(cublasSgetriBatched(handle,n,(const float**)A_d,lda,P,C_d,lda,INFO,batchSize));

    cudacall(cudaMemcpy(&INFOh,INFO,sizeof(int),cudaMemcpyDeviceToHost));

    if(INFOh != 0)
    {
        fprintf(stderr, "Inversion Failed: Matrix is singular\n");
        cudaDeviceReset();
        exit(EXIT_FAILURE);
    }

    cudaFree(P);
    cudaFree(INFO);
    cublasDestroy_v2(handle);

    cudacall(cudaMemcpy(dst,dst_d,n * n * sizeof(float),cudaMemcpyDeviceToHost));

    cudaFree(src_d);
    cudaFree(dst_d);
    cudaFree(C_d);
    cudaFree(A_d);
}
void multicublas(float* src1, float*src2,float* dst, int n1,int n2,int n3)
{
    float* src_d1,*src_d2, *dst_d;

    cudacall(cudaMalloc((void**)&src_d1,n1 * n2 * sizeof(float)));
    cudacall(cudaMemcpy(src_d1,src1,n1 * n2 * sizeof(float),cudaMemcpyHostToDevice));
    cudacall(cudaMalloc((void**)&src_d2,n2 * n3 * sizeof(float)));
    cudacall(cudaMemcpy(src_d2,src2,n2 * n3 * sizeof(float),cudaMemcpyHostToDevice));
    cudacall(cudaMalloc((void**)&dst_d,n3 * n1 * sizeof(float)));
    cudacall(cudaMemset(dst_d,0,n3 * n1 * sizeof(float)));
    /*float* C = dst;
    cudacall(cudaMalloc<float>(&C,n * n * sizeof(float)));*/
    cublasHandle_t handle;
    cublascall(cublasCreate_v2(&handle));

    float alpha=1.0;
    float beta=0.0;

    cublasSgemm(handle,CUBLAS_OP_N,CUBLAS_OP_N,n3,n1,n2,&alpha,src_d2,n3,src_d1,n2,&beta,dst_d,n3);
    cudacall(cudaMemcpy(dst,dst_d,n3 * n1 * sizeof(float),cudaMemcpyDeviceToHost));
//printf("%f",*(dst+2));
    cublasDestroy_v2(handle);

    cudaFree(src_d1);
    cudaFree(src_d2);
    cudaFree(dst_d);

}
int main()
{
    const int m=3;
    const int n = 3;
    const int l = 1;

       //Random matrix with full pivots
    float full_pivots[m*n] = { 1, 2, 3,
                               4, 5, 6,
                               7, 8, 1 };
    float full_pivots1[n*l] = { 1, 2, 3};
    /*float full_pivots2[m*n] = { 1+0.007*0.007, 0, 0, 0, 0, 0,
                               0, 1+0.008*0.008, 0, 0, 0, 0,
                               0, 0, 1+0.008*0.008, 0, 0, 0,
                               0, 0, 0, 1+0.017*0.017, 0, 0,
                               0, 0, 0, 0, 1+0.05*0.05, 0,
                               0, 0, 0, 0, 0, 1+0.141*0.141};*/
    float* a = full_pivots;
    float* a1 = full_pivots1;
    //float* a2 = full_pivots2;
    invert(a,a,n);
    //multicublas(a,a1,a1,m,n,l);
    for(int i=0; i<m; i++)
    {
        for(int j=0; j<n; j++)
        fprintf(stdout,"%f\t",a[i*n+j]);
        fprintf(stdout,"\n");
    }
    return 0;
}
