#include "../include/common.h"
#include <cuda_runtime.h>
#include <stdio.h>

// this to avoid the overhead of first run
__global__ void warmingUp(float *c)
{
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    float ia, ib;
    ia = ib = 0.0f;

    if (tid % 2 == 0)
    {
        ia = 200.0f;
    }
    else
    {
        ib = 100.0f;
    }
    c[tid] = ia + ib;
}

__global__ void kernel1(float *c)
{
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    float ia, ib;
    ia = ib = 0.0f;

    if (tid % 2 == 0)
    {
        ia = 200.0f;
    }
    else
    {
        ib = 100.0f;
    }
    c[tid] = ia + ib;
}

__global__ void kernel2(float *c)
{
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    float ia, ib;
    ia = ib = 0.0f;

    if ((tid / warpSize) % 2 == 0)
    {
        ia = 200.0f;
    }
    else
    {
        ib = 100.0f;
    }
    c[tid] = ia + ib;
}

__global__ void kernel3(float *c)
{
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    float ia, ib;
    ia = ib = 0.0f;
    bool pred = (tid % 2 == 0);
    if (pred)
    {
        ia = 200.f;
    }
    if(!pred)
    {
        ib = 100.0f;
    }
    c[tid] = ia + ib;
}

int main(int argc, char *argv[])
{
    // setup the device
    int dev = 0;
    cudaDeviceProp deviceProp;
    CHECK(cudaGetDeviceProperties(&deviceProp, dev));
    printf("%s using device %d : %s\n", argv[0], dev, deviceProp.name);

    // set time measure
    // cudaEvent_t start, stop;
    // float elaspedTime;
    // cudaEventCreate(&start);
    // cudaEventCreate(&stop);

    // set up data size
    int size = atoi(argv[1]);
    printf("Data size : %d \n", size);

    // execution configuration
    dim3 block(atoi(argv[2]));
    dim3 grid((size + block.x - 1) / block.x);
    printf("Execution configuration <<<%d,%d>>> \n", grid.x, block.x);

    // allocate the gpu memory
    float *d_C;
    size_t nBytes = size * sizeof(float);
    cudaMalloc((float **)&d_C, nBytes);
    cudaMemset(d_C, 0, size);

    // host memory
    float *h_C;
    h_C = (float *)malloc(nBytes);
    memset(h_C, 0, size);

    // run the warmup kernel to remove the overhead
    // cudaEventRecord(start,0);
    warmingUp<<<grid, block>>>(d_C);
    // cudaEventRecord(stop,0);
    // CHECK(cudaEventSynchronize(stop));

    // cudaEventElapsedTime(&elaspedTime, start,stop);
    // printf("Kernel 2       <<<%d,%d>>> elapsed %f ms \n",
    //        grid.x, block.x, elaspedTime);
    // CHECK(cudaGetLastError());

    // run  kernel 1
    // cudaEventRecord(start,0);
    kernel1<<<grid, block>>>(d_C);
    // cudaEventRecord(stop,0);
    // CHECK(cudaEventSynchronize(stop));

    // cudaEventElapsedTime(&elaspedTime, start,stop);
    // printf("Kernel 2       <<<%d,%d>>> elapsed %f ms \n",
    //        grid.x, block.x, elaspedTime);
    // CHECK(cudaGetLastError());

    // run  kernel 2
    // cudaEventRecord(start,0);
    kernel2<<<grid, block>>>(d_C);
    // cudaEventRecord(stop,0);
    // CHECK(cudaEventSynchronize(stop));

    // cudaEventElapsedTime(&elaspedTime, start,stop);
    // printf("Kernel 2       <<<%d,%d>>> elapsed %f ms \n",
    //        grid.x, block.x, elaspedTime);
    // CHECK(cudaGetLastError());

    kernel3<<<grid, block>>>(d_C);

    // Free Gpu memory
    CHECK(cudaFree(d_C));
    CHECK(cudaDeviceReset());
    return EXIT_SUCCESS;
}
