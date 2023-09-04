#include "../include/common.h"
#include <cuda_runtime.h>
#include <stdio.h>

void checkResult(float *hostRef, float *gpuRef, const unsigned int N)
{
    double eps = 1.0E-8;
    for (unsigned int i = 0; i < N; i++)
    {
        if (abs(hostRef[i] - gpuRef[i]) > eps)
        {
            printf("Array does not match\n");
            printf("host %5.2f gpu %5.2f at current %d\n", hostRef[i], gpuRef[i], i);
            break;
        }
    }
}

void initialData(float *A, const unsigned int N)
{
    for (unsigned int i = 0; i < N; i++)
    {
        A[i] = (float)(rand() & 0xFF) / 10.0f;
    }
}

void sumArrayOnHost(float *A, float *B, float *C, const unsigned int N)
{
    for (unsigned int i = 0; i < N; i++)
    {
        C[i] = A[i] + B[i];
    }
}

__global__ void sumArrays(float *A, float *B, float *C, const unsigned int N)
{
    unsigned int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < N)
        C[idx] = A[idx] + B[idx];
}

__global__ void sumArraysZeroCopy(float *A, float *B, float *C, const unsigned int N)
{
    unsigned int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < N)
        C[idx] = A[idx] + B[idx];
}

int main(int argc, char *argv[])
{
    // set up device
    int dev = 0;
    CHECK(cudaSetDevice(dev));

    cudaDeviceProp devProp;
    CHECK(cudaGetDeviceProperties(&devProp, dev));

    // check for support pinned memory
    if (!devProp.canMapHostMemory)
    {
        printf("Device %d does not support mapping CPU Host memory!\n", dev);
        CHECK(cudaDeviceReset());
        exit(EXIT_FAILURE);
    }

    printf("Using the device %d : %s\n", dev, devProp.name);

    // set up data size of vector
    int pow = 24;

    if (argc > 1)
        pow = atoi(argv[1]);

    int nElem = 1 << pow;
    size_t nBytes = nElem * sizeof(float);

    if (pow < 18)
    {
        printf("Vector size %d power %d  nbytes  %3.0f KB\n", nElem, pow,
               (float)nBytes / (1024.0f));
    }
    else
    {
        printf("Vector size %d power %d  nbytes  %3.0f MB\n", nElem, pow,
               (float)nBytes / (1024.0f * 1024.0f));
    }

    // Using deivce memory

    // Host Memory
    float *h_A, *h_B, *hostRef, *gpuRef;
    h_A = (float *)malloc(nBytes);
    h_B = (float *)malloc(nBytes);
    hostRef = (float *)malloc(nBytes);
    gpuRef = (float *)malloc(nBytes);

    // init data
    initialData(h_A, nElem);
    initialData(h_B, nElem);
    memset(hostRef, 0, nBytes);
    memset(gpuRef, 0, nBytes);

    // add vector on cpu side
    sumArrayOnHost(h_A, h_B, hostRef, nElem);

    // deivce side
    float *d_A, *d_B, *d_C;
    CHECK(cudaMalloc((float **)&d_A, nBytes));
    CHECK(cudaMalloc((float **)&d_B, nBytes));
    CHECK(cudaMalloc((float **)&d_C, nBytes));

    // transfer the data HOST to DEVICE
    CHECK(cudaMemcpy(d_A, h_A, nBytes, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_B, h_B, nBytes, cudaMemcpyHostToDevice));

    // kernel configuration
    dim3 block(512);
    dim3 grid((nElem + block.x - 1) / block.x);

    // launch the kernel
    sumArrays<<<grid, block>>>(d_A, d_B, d_C, nElem);

    // get back the results
    CHECK(cudaMemcpy(gpuRef, d_C, nBytes, cudaMemcpyDeviceToHost));

    // check the result
    checkResult(hostRef, gpuRef, nElem);

    // free the memory cpu anfd gpu
    cudaFree(d_A);
    cudaFree(d_B);
    free(h_A);
    free(h_B);

    // Using the zero copy its mapped memory no explict transfer of data
    // from host to device

    // allocate the zero copy memory
    // return the host memory that mapped into device address space
    printf("USING ZERO COPY\n");
    int flags = cudaHostAllocMapped;
    cudaHostAlloc((void **)&h_A, nBytes, flags);
    cudaHostAlloc((void **)&h_B, nBytes, flags);

    // init data
    initialData(h_A, nElem);
    initialData(h_B, nElem);
    memset(hostRef, 0, nBytes);
    memset(hostRef, 0, nBytes);

    // get the data pointer of HOST and passing to device
    CHECK(cudaHostGetDevicePointer((void **)&d_A, (void *)h_A, 0));
    CHECK(cudaHostGetDevicePointer((void **)&d_B, (void *)h_B, 0));

    sumArrayOnHost(h_A, h_B, hostRef, nElem);

    // launch the kernel but data in host mapped memory
    sumArraysZeroCopy<<<grid, block>>>(d_A, d_B, d_C, nElem);

    cudaMemcpy(gpuRef, d_C, nBytes, cudaMemcpyDeviceToHost);
    checkResult(hostRef, gpuRef, nElem);

    // free the gpu and cpu memory
    cudaFree(d_C);
    cudaFreeHost(h_A);
    cudaFreeHost(h_B);

    free(hostRef);
    free(gpuRef);

    cudaDeviceReset();
    return EXIT_SUCCESS;
}