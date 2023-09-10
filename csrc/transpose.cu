#include "../include/common.h"
#include <ATen/ATen.h>
#include <string>
#include <iostream>

__global__ void warmup(const float *__restrict__ A, float *C, const int nx, const int ny)
{
    unsigned int ix = blockDim.x * blockIdx.x + threadIdx.x;
    unsigned int iy = blockDim.y * blockIdx.y + threadIdx.y;
    if (ix < nx && iy < ny)
    {
        C[iy * nx + ix] = A[iy * nx + ix];
    }
}

__global__ void copyRow(const float *__restrict__ A, float *C, const int nx, const int ny)
{
    unsigned int ix = blockDim.x * blockIdx.x + threadIdx.x;
    unsigned int iy = blockDim.y * blockIdx.y + threadIdx.y;
    if (ix < nx && iy < ny)
    {
        C[iy * nx + ix] = A[iy * nx + ix] + 10;
    }
}

__global__ void copyCol(const float *__restrict__ A, float *C, const int nx, const int ny)
{
    unsigned int ix = blockDim.x * blockIdx.y + threadIdx.x;
    unsigned int iy = blockDim.y * blockIdx.y + threadIdx.y;
    if (ix < nx && iy < ny)
    {
        C[ix * ny + iy] = A[ix * ny + iy];
    }
}

int main(int argc, char *argv[])
{
    // set the array size
    int nx = 1 << 11;
    int ny = 1 << 11;
    size_t nBytes = nx * ny * sizeof(float);
    std::cout << "Tensor size (" << nx << "," << ny << ") and " << nBytes << "  bytes." << std::endl;

    // allocate on host
    at::Tensor h_A = at::rand({nx, ny}, at::kFloat);
    at::Tensor h_C = at::zeros_like(h_A);
    // std::cout << h_A << std::endl;
    // std::cout << h_C << std::endl;
    //  allocate on device
    float *d_A, *d_C;
    CHECK_ERROR(cudaMalloc((float **)&d_A, nBytes));
    CHECK_ERROR(cudaMalloc((float **)&d_C, nBytes));

    // transfer host to device
    CHECK_ERROR(cudaMemcpy(d_A, h_A.data_ptr(), nBytes, cudaMemcpyHostToDevice));

    // execution configuration
    int iKernel = std::stoi(argv[1]);
    int blockX = std::stoi(argv[2]);
    int blockY = std::stoi(argv[3]);

    dim3 block(blockX, blockY);
    dim3 grid((nx + block.x - 1) / block.x, (ny + block.y - 1) / block.y);

    // warmup kernel
    // warmup<<<grid, block>>>(d_A, d_C, nx, ny);
    // CHECK_ERROR(cudaDeviceSynchronize());

    // different kernel use the func pointer and switch
    void (*kernel)(const float *, float *, const int, const int);
    std::string kernelName;

    switch (iKernel)
    {
    case 0:
        kernel = &copyRow;
        kernelName = "CopyRow           ";
        break;
    case 1:
        kernel = &copyCol;
        kernelName = "CopyCol           ";
        break;
    }

    kernel<<<grid, block>>>(d_A, d_C, nx, ny);
    CHECK_ERROR(cudaDeviceSynchronize());

    CHECK_ERROR(cudaMemcpy(h_C.data_ptr(), d_C, nBytes, cudaMemcpyDeviceToHost));

    if (iKernel > 1)
    {
        if (h_A.transpose(1, 0).allclose(h_C))
        {
            std::cout << "Tensor Match" << std::endl;
        }
        else
        {
            std::cout << "Tensor Not Match" << std::endl;
        }
    }
    else
    {
        if ((h_A + 10 ).allclose(h_C))
        {
            std::cout << "Tensor Match" << std::endl;
        }
        else
        {
            std::cout << "Tensor Not Match" << std::endl;
        }
    }
    // std::cout << (h_A + 10 )<< std::endl;
    // std::cout << h_C << std::endl;

    CHECK_ERROR(cudaFree(d_A));
    CHECK_ERROR(cudaFree(d_C));
    CHECK_ERROR(cudaDeviceReset());
    return EXIT_SUCCESS;
}