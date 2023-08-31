#include <cuda_runtime.h>
#include <stdio.h>

#define CHECK(call)                                                             \
    {                                                                           \
        const cudaError_t error = call;                                         \
        if (error != cudaSuccess)                                               \
        {                                                                       \
            printf("Error: %s:%d, ", __FILE__, __LINE__);                       \
            printf("code:%d ,reason: %s \n", error, cudaGetErrorString(error)); \
            exit(1);                                                            \
        }                                                                       \
    }

void checkResult(const float *hostRef, const float *gpuRef, const int N)
{
    double eps = 1.0E-8;
    bool match = 1;
    for (int i = 0; i < N; i++)
    {
        if (abs(hostRef[i] - gpuRef[i]) > eps)
        {
            match = 0;
            printf("Array do not match! \n");
            printf("host %5.2f gpu %5.2f af current %d \n", hostRef[i], gpuRef[i], i);
            break;
        }
    }
    if (match)
        printf("Array match. \n\n");
}

void initialData(float *ip, int size)
{
    time_t t;
    srand((unsigned)time(&t));

    for (int i = 0; i < size; i++)
    {
        ip[i] = (float)(rand() & 0xFF) / 10.0f;
    }
}

void sumArrayOnHost(const float *A, const float *B, float *C, const int N)
{
    for (int i = 0; i < N; i++)
    {
        C[i] = A[i] + B[i];
    }
}

__global__ void sumArrayOnGPU(float *A, float *B, float *C, const int N)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N)
        C[i] = A[i] + B[i];
}

int main(int argc, char *argv[])
{
    // set the device
    int device = 0;
    cudaSetDevice(device);

    // set the data size
    unsigned long int nElem = atoi(argv[1]);
    printf("Vector size %ld\n", nElem);

    // number of bytes
    size_t nBytes = nElem * sizeof(float);

    // allocate host memory
    float *h_A, *h_B, *hostRef, *gpuRef;
    h_A = (float *)malloc(nBytes);
    h_B = (float *)malloc(nBytes);
    hostRef = (float *)malloc(nBytes);
    gpuRef = (float *)malloc(nBytes);

    // initialize the data
    initialData(h_A, nElem);
    initialData(h_B, nElem);

    memset(hostRef, 0, nElem);
    memset(gpuRef, 0, nElem);

    // allocat the device memory
    float *d_A, *d_B, *d_C;
    CHECK(cudaMalloc((float **)(&d_A), nBytes));
    CHECK(cudaMalloc((float **)(&d_B), nBytes));
    CHECK(cudaMalloc((float **)(&d_C), nBytes));

    // transfer the data host to device
    CHECK(cudaMemcpy(d_A, h_A, nBytes, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_B, h_B, nBytes, cudaMemcpyHostToDevice));

    // invoke kernel
    dim3 block(atoi(argv[2]));
    dim3 grid((nElem + block.x - 1) / block.x);

    sumArrayOnGPU<<<grid, block>>>(d_A, d_B, d_C, nElem);
    CHECK(cudaDeviceSynchronize());
    printf("Execution Configuration <<<%d,%d>>>\n", grid.x, block.x);

    // copy back the result
    CHECK(cudaMemcpy(gpuRef, d_C, nBytes, cudaMemcpyDeviceToHost);)

    // add vecotr in host side
    sumArrayOnHost(h_A, h_B, hostRef, nElem);

    checkResult(hostRef, gpuRef, nElem);

    // free the device memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    // free host memory
    free(h_A);
    free(h_B);
    free(hostRef);
    free(gpuRef);

    return (0);
}