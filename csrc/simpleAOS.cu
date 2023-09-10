#include "../include/common.h"
#include <cuda_runtime.h>
#include <stdio.h>

#define LEN (1 << 22)

struct innerArray
{
    float x;
    float y;
};

void initialInnerArray(innerArray *p, const unsigned int N)
{
    for (unsigned int i = 0; i < N; i++)
    {
        p[i].x = (float)(rand() & 0xFF) / 10.f;
        p[i].y = (float)(rand() & 0xFF) / 10.f;
    }
}

void testInnerStructHost(const innerArray *A, innerArray *B, const unsigned int N)
{
    for (unsigned int i = 0; i < N; i++)
    {
        B[i].x = A[i].x + 10.0f;
        B[i].y = A[i].y + 20.0f;
    }
}

void checkResult(const innerArray *hostRef, const innerArray *gpuRef, const unsigned int N)
{
    double eps = 1.0E-8;
    for (unsigned int i = 0; i < N; i++)
    {
        if ((abs(hostRef[i].x - gpuRef[i].x) > eps) || (abs(hostRef[i].y - gpuRef[i].y) > eps))
        {
            printf("Array do not match\n");
            break;
        }
    }
    printf("Array match\n");
}

__global__ void testInnerStruct(const innerArray *__restrict__ A, innerArray *B, const unsigned int N)
{
    unsigned int tid = blockDim.x * blockIdx.x + threadIdx.x;
    if (tid < N)
    {
        B[tid].x = A[tid].x + 10.0f;
        B[tid].y = A[tid].y + 20.0f;
    }
}

int main(int argc, char *argv[])
{
    // number of element and size
    unsigned int nElem = LEN;
    unsigned int nBytes = LEN * sizeof(innerArray);
    printf("Array with %d elems and %d\n",nElem,nBytes);

    // allocate on host
    innerArray *h_A, *hostRef, *gpuRef;
    h_A = (innerArray *)malloc(nBytes);
    hostRef = (innerArray *)malloc(nBytes);
    gpuRef = (innerArray *)malloc(nBytes);

    // initiall data and host side calculation
    initialInnerArray(h_A, nElem);
    testInnerStructHost(h_A, hostRef, nElem);

    // allocate on device memory
    innerArray *d_A, *d_B;
    CHECK_ERROR(cudaMalloc((innerArray **)&d_A, nBytes));
    CHECK_ERROR(cudaMalloc((innerArray **)&d_B, nBytes));

    // copy data from host to device
    CHECK_ERROR(cudaMemcpy(d_A, h_A, nBytes, cudaMemcpyHostToDevice));

    // execution configuration
    int blockSize = 512;
    if (argc > 1)
        blockSize = atoi(argv[1]);

    dim3 block(blockSize);
    dim3 grid((nElem + block.x - 1) / block.x);

    // launch the kernel
    printf("Launching testInnerStruct<<<%d,%d>>> \n", grid.x, block.x);
    testInnerStruct<<<grid, block>>>(d_A, d_B, nElem);
    CHECK_ERROR(cudaDeviceSynchronize());

    // copy back the result
    CHECK_ERROR(cudaMemcpy(gpuRef, d_B, nBytes, cudaMemcpyDeviceToHost));

    checkResult(hostRef, gpuRef, nElem);

    // free memory on host and device
    cudaFree(d_A);
    cudaFree(d_B);
    free(h_A);
    free(hostRef);
    free(gpuRef);

    CHECK_ERROR(cudaDeviceReset());
    return EXIT_SUCCESS;
}
