#include "../include/common.h"
#include <cuda_runtime.h>
#include <stdio.h>

#define LEN (1 << 22)

struct innerArray
{
    float x[LEN];
    float y[LEN];
};

void initialInnerArray(innerArray *p, const unsigned int N)
{
    for (unsigned int i = 0; i < N; i++)
    {
        p->x[i] = (float)(rand() & 0xFF) / 10.f;
        p->y[i] = (float)(rand() & 0xFF) / 10.f;
    }
}

void testInnerStructHost(const innerArray *A, innerArray *B, const unsigned int N)
{
    for (unsigned int i = 0; i < N; i++)
    {
        B->x[i] = A->x[i] + 10.0f;
        B->y[i] = A->y[i] + 20.0f;
    }
}

void checkResult(const innerArray *hostRef, const innerArray *gpuRef, const unsigned int N)
{
    double eps = 1.0E-8;
    for (unsigned int i = 0; i < N; i++)
    {
        if ((abs(hostRef->x[i] - gpuRef->x[i]) > eps) || (abs(hostRef->y[i] - gpuRef->y[i]) > eps))
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
        B->x[tid] = A->x[tid] + 10.0f;
        B->y[tid] = A->y[tid] + 20.0f;
    }
}

int main(int argc, char *argv[])
{
    // number of element and size
    unsigned int nElem = LEN;
    unsigned int nBytes = sizeof(innerArray);
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
    CHECK(cudaMalloc((innerArray **)&d_A, nBytes));
    CHECK(cudaMalloc((innerArray **)&d_B, nBytes));

    // copy data from host to device
    CHECK(cudaMemcpy(d_A, h_A, nBytes, cudaMemcpyHostToDevice));

    // execution configuration
    int blockSize = 512;
    if (argc > 1)
        blockSize = atoi(argv[1]);

    dim3 block(blockSize);
    dim3 grid((nElem + block.x - 1) / block.x);

    // launch the kernel
    printf("Launching testInnerStruct<<<%d,%d>>> \n", grid.x, block.x);
    testInnerStruct<<<grid, block>>>(d_A, d_B, nElem);
    CHECK(cudaDeviceSynchronize());

    // copy back the result
    CHECK(cudaMemcpy(gpuRef, d_B, nBytes, cudaMemcpyDeviceToHost));

    checkResult(hostRef, gpuRef, nElem);

    // free memory on host and device
    cudaFree(d_A);
    cudaFree(d_B);
    free(h_A);
    free(hostRef);
    free(gpuRef);

    CHECK(cudaDeviceReset());
    return EXIT_SUCCESS;
}
