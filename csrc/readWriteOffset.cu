#include "../include/common.h"
#include <cuda_runtime.h>
#include <stdio.h>
#include <string.h>

void initialData(float *A, const unsigned long int N)
{
    for (unsigned long int i = 0; i < N; i++)
    {
        A[i] = (float)(rand() & 0xFF) / 10.0f;
    }
}

__global__ void readOffset(float *A, float *B, float *C, const unsigned long int N, const int offset)
{
    unsigned long int tid = blockDim.x * blockIdx.x + threadIdx.x;
    unsigned long int k = tid + offset;
    if (k < N)
        C[tid] = A[k] + B[k];
}

__global__ void readOnlyCache(float *A, float *B, float *C, const unsigned long int N, const int offset)
{
    unsigned long tid = blockDim.x * blockIdx.x + threadIdx.x;
    unsigned long int k = tid + offset;
    if (k < N)
        C[tid] = __ldg(&A[k]) + __ldg(&B[k]);
}

int main(int argc, char *argv[])
{
    // set up the device
    int dev = 0;
    cudaDeviceProp devProp;

    CHECK_ERROR(cudaSetDevice(dev));
    CHECK_ERROR(cudaGetDeviceProperties_v2(&devProp, dev));
    printf("Using device %d : %s\n", dev, devProp.name);

    // set up array size
    unsigned long int nElems = 1 << 24;
    size_t nBytes = nElems * sizeof(float);
    printf("Array size %ld of %zu bytes\n", nElems, nBytes);

    // set block size and offset
    // expect offset format 0,11,128
    char *input = argv[1];
    int blockSize = atoi(argv[2]);

    // execution configuration
    dim3 block(blockSize);
    dim3 grid((nElems + block.x - 1) / block.x);

    // allocate the host memory
    float *h_A, *h_B;
    h_A = (float *)malloc(nBytes);
    h_B = (float *)malloc(nBytes);

    // initial data
    initialData(h_A, nElems);
    initialData(h_B, nElems);

    // allocate the device memory
    float *d_A, *d_B, *d_C;
    CHECK_ERROR(cudaMalloc((void **)&d_A, nBytes));
    CHECK_ERROR(cudaMalloc((void **)&d_B, nBytes));
    CHECK_ERROR(cudaMalloc((void **)&d_C, nBytes));

    // transfer data from host to device
    CHECK_ERROR(cudaMemcpy(d_A, h_A, nBytes, cudaMemcpyHostToDevice));
    CHECK_ERROR(cudaMemcpy(d_B, h_B, nBytes, cudaMemcpyHostToDevice));

    char *token;
    int offset;
    token = strtok(input, ",");
    while (token != NULL)
    {
        offset = atoi(token);
        // launch the kernel 1 : Read Offset
        printf("Launch Kernel<<<%d,%d>>> with %d offset\n", grid.x, block.x, offset);
        readOffset<<<grid, block>>>(d_A, d_B, d_C, nElems, offset);
        CHECK_ERROR(cudaDeviceSynchronize());

        // Launch Kernel 2 : Read only Cache
        printf("Launch Kernel<<<%d,%d>>> with %d offset\n", grid.x, block.x, offset);
        readOnlyCache<<<grid, block>>>(d_A, d_B, d_C, nElems, offset);
        CHECK_ERROR(cudaDeviceSynchronize());

        token = strtok(NULL, ",");
    }

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    free(h_A);
    free(h_B);

    CHECK_ERROR(cudaDeviceReset());
    return EXIT_SUCCESS;
}