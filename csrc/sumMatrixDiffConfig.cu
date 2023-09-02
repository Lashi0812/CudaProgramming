#include "../include/common.h"
#include <cuda_runtime.h>
#include <stdio.h>

void initialData(float *ip, unsigned int size)
{
    printf("Init");
    for (unsigned int i = 0; i < size; i++)
    {
        ip[i] = (float)(rand() & 0xFF) / 10.0f;
    }
}

void sumMatrixOnHost(float *A, float *B, float *C, const unsigned int nx, const unsigned int ny)
{
    float *ia = A;
    float *ib = B;
    float *ic = C;

    for (unsigned int iy = 0; iy < ny; iy++)
    {
        for (unsigned int ix = 0; ix < nx; ix++)
        {
            ic[ix] = ia[ix] + ib[ix];
        }

        ia += nx;
        ib += nx;
        ic += nx;
    }

    return;
}

void checkResult(float *hostRef, float *gpuRef, const int N)
{
    double epsilon = 1.0E-8;

    for (long long int i = 0; i < N; i++)
    {
        if (abs(hostRef[i] - gpuRef[i]) > epsilon)
        {
            printf("host %f gpu %f ", hostRef[i], gpuRef[i]);
            printf("Arrays do not match.\n\n");
            break;
        }
    }
}

__global__ void sumMatrixOnGPU2D(const float *A, const float *B, float *C, const unsigned int NX, const unsigned int NY)
{
    unsigned int ix = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int iy = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned int idx = iy * NX + ix;

    if (ix < NX && iy < NY)
    {
        C[idx] = A[idx] + B[idx];
    }
}

int main(int argc, char *argv[])
{   
    // setup the device
    int dev = 0;
    cudaDeviceProp deviceProp;
    CHECK(cudaGetDeviceProperties(&deviceProp, dev));
    printf("%s using device %d : %s\n", argv[0], dev, deviceProp.name);

    unsigned int nx = 1 << 15;
    unsigned int ny = 1 << 15;

    unsigned int nxy = nx * ny;
    size_t nBytes = nxy * sizeof(float);

    // malloc host memory
    float *h_A, *h_B, *hostRef, *gpuRef;
    h_A = (float *)malloc(nBytes);
    h_B = (float *)malloc(nBytes);
    hostRef = (float *)malloc(nBytes);
    gpuRef = (float *)malloc(nBytes);

    // initial data
    printf("Initailizing data for (%d,%d) total %d of %ld bytes....\n",nx,ny,nxy,nBytes);
    initialData(h_A,nxy);
    initialData(h_B,nxy);

    memset(hostRef, 0, nBytes);
    memset(gpuRef, 0, nBytes);

    sumMatrixOnHost(h_A, h_B, hostRef, nx, ny);
    

    // malloc gpu
    float *d_MatA, *d_MatB, *d_MatC;
    CHECK(cudaMalloc((void **)&d_MatA, nBytes));
    CHECK(cudaMalloc((void **)&d_MatB, nBytes));
    CHECK(cudaMalloc((void **)&d_MatC, nBytes));

    // tranfer data from host to device
    CHECK(cudaMemcpy(d_MatA, h_A, nBytes, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_MatB, h_B, nBytes, cudaMemcpyHostToDevice));

    dim3 blockConfigs[] = {
        dim3(1024,1),
        dim3(512,2),
        dim3(512,1),
        dim3(256,4),
        dim3(256,2),
        dim3(256,1),
        dim3(128,8),
        dim3(128,4),
        dim3(128,2),
        dim3(128,1),
        dim3(64,16),
        dim3(64,8),
        dim3(64,4),
        dim3(64,2),
        dim3(64,1),
        dim3(32,32),
        dim3(32,16),
        dim3(32,8),
        dim3(32,4),
        dim3(32,2),
        dim3(32,1),
        dim3(16,16),
        dim3(16,8),
        dim3(16,4),
        dim3(16,2),
        dim3(16,1),
        dim3(8,8),
        dim3(8,4),
        dim3(8,2),
        dim3(8,1),
        dim3(4,4),
        dim3(4,2),
        dim3(4,1),
        dim3(2,2),
        dim3(2,1),
    };

    for (int i = 0; i < sizeof(blockConfigs) / sizeof(blockConfigs[0]); ++i) {

        dim3 block = blockConfigs[i];
        dim3 grid((nx + block.x - 1) / block.x, (ny + block.y - 1) / block.y);

        printf("sumMatrixOnGPU2D <<<(%d,%d), (%d,%d)>>>\n", grid.x,
            grid.y,
            block.x, block.y);
        
        CHECK(cudaDeviceSynchronize());
        sumMatrixOnGPU2D<<<grid, block>>>(d_MatA, d_MatB, d_MatC, nx, ny);
        CHECK(cudaDeviceSynchronize());
        printf("sumMatrixOnGPU2D <<<(%d,%d), (%d,%d)>>>\n", grid.x,
            grid.y,
            block.x, block.y);

        CHECK(cudaMemcpy(gpuRef, d_MatC, nBytes, cudaMemcpyDeviceToHost));

        // check the results
        checkResult(hostRef, gpuRef, nxy);
    }
    // free gpu memory
    CHECK(cudaFree(d_MatA));
    CHECK(cudaFree(d_MatB));
    CHECK(cudaFree(d_MatC));

    // free host memory
    free(h_A);
    free(h_B);
    free(hostRef);
    free(gpuRef);

    CHECK(cudaDeviceReset());

    return EXIT_SUCCESS;
}