#include "../include/common.h"
#include <cuda_runtime.h>
#include <ATen/ATen.h>
#include <iostream>

#define BDIMX 32
#define BDIMY 16
// since 32 thread in warp need 2 col each of 16
#define IPAD 2

__global__ void setRowReadRow(int *out)
{
    // allocate the static shared mem
    // inner dim of shared mem should match the inner dim of thread block

    __shared__ int tile[BDIMY][BDIMX];

    // get the global thread id
    unsigned int tid = threadIdx.y * blockDim.x + threadIdx.x;

    // store in row-wise
    tile[threadIdx.y][threadIdx.x] = tid;

    __syncthreads();

    // load in row wise
    out[tid] = tile[threadIdx.y][threadIdx.x];
}

__global__ void setColReadCol(int *out)
{
    // allocate static shared memory
    // inner most col shared mem should match the outer dim of thread block
    __shared__ int tile[BDIMX][BDIMY];

    // get the global thread id
    unsigned int tid = threadIdx.y * blockDim.x + threadIdx.x;

    // store in col wise
    tile[threadIdx.x][threadIdx.y] = tid;

    __syncthreads();

    // load in col wise
    out[tid] = tile[threadIdx.x][threadIdx.y];
}

__global__ void setRowReadCol(int *out)
{
    // allocate the static shared mem
    __shared__ int tile[BDIMY][BDIMX];

    // global tid
    unsigned int gid = threadIdx.y * blockDim.x + threadIdx.x;

    // convert the gid into transpose (row,col)
    unsigned int irow = gid / blockDim.y;
    unsigned int icol = gid % blockDim.y;

    // store row wise
    tile[threadIdx.y][threadIdx.x] = gid;

    __syncthreads();

    // Load col wise
    out[gid] = tile[icol][irow];
}

__global__ void setRowReadColDyn(int *out)
{
    // declare dynamic shared array
    extern __shared__ int tile[];

    // get the global thread id
    unsigned int gid = threadIdx.y * blockDim.x + threadIdx.x;

    // convert gid into transpose (row,col)
    unsigned int irow = gid / blockDim.y;
    unsigned int icol = gid % blockDim.y;

    // convert back to 1D array idx
    unsigned int col_idx = icol * blockDim.x + irow;

    // store in row-wise
    tile[gid] = gid;

    __syncthreads();

    // load in col-wise
    out[gid] = tile[col_idx];
}

__global__ void setRowReadColPad(int *out)
{
    // allocate the static shared mem
    __shared__ int tile[BDIMY][BDIMX + IPAD];

    // global tid
    unsigned int gid = threadIdx.y * blockDim.x + threadIdx.x;

    // convert the gid into transpose (row,col)
    unsigned int irow = gid / blockDim.y;
    unsigned int icol = gid % blockDim.y;

    // store row wise
    tile[threadIdx.y][threadIdx.x] = gid;

    __syncthreads();

    // Load col wise
    out[gid] = tile[icol][irow];
}

__global__ void setRowReadColDynPad(int *out)
{
    // declare dynamic shared array
    extern __shared__ int tile[];

    // get the global thread id
    unsigned int gid = threadIdx.y * blockDim.x + threadIdx.x;

    // convert gid into transpose (row,col)
    unsigned int irow = gid / blockDim.y;
    unsigned int icol = gid % blockDim.y;

    unsigned int row_idx = threadIdx.y * (blockDim.x + IPAD) + threadIdx.x;

    // convert back to 1D array idx
    unsigned int col_idx = icol * (blockDim.x + IPAD) + irow;

    // store in row-wise
    tile[row_idx] = gid;

    __syncthreads();

    // load in col-wise
    out[gid] = tile[col_idx];
}

int main(int argc, char *argv[])
{
    // setup device
    int dev = 0;
    cudaDeviceProp deviceProp;
    cudaSharedMemConfig smemConfig;

    CHECK_ERROR(cudaGetDeviceProperties(&deviceProp, dev));
    CHECK_ERROR(cudaSetDevice(dev));
    CHECK_ERROR(cudaDeviceGetSharedMemConfig(&smemConfig));

    std::cout << "Using device " << dev << " : " << deviceProp.name
              << " with smem config " << smemConfig
              << " of bank mode " << (smemConfig == 1 ? " 4-bytes" : "8-bytes")
              << std::endl;

    int nx = BDIMX;
    int ny = BDIMY;
    unsigned int nBytes = nx * ny * sizeof(int);
    bool iprint = 0;
    if (argc > 1)
        iprint = std::stoi(argv[1]);

    // host side
    at::Tensor gpuRef = at::zeros({nx, ny}, at::kInt);

    // device side
    int *d_C;
    CHECK_ERROR(cudaMalloc((int **)&d_C, nBytes));

    // execution configuration
    dim3 block(nx, ny);
    dim3 grid(1, 1);

    std::cout << "Launching with <<<(" << grid.x << "," << grid.y << "),(" << block.x << "," << block.y << ")>>>" << std::endl;

    // launch the kernel setRowReadRow
    CHECK_ERROR(cudaMemset(d_C, 0, nBytes));
    setRowReadRow<<<grid, block>>>(d_C);
    CHECK_ERROR(cudaMemcpy(gpuRef.data_ptr(), d_C, nBytes, cudaMemcpyDeviceToHost));
    if (iprint)
        std::cout << gpuRef << std::endl;

    // launch the kernel setColReadCol
    CHECK_ERROR(cudaMemset(d_C, 0, nBytes));
    setColReadCol<<<grid, block>>>(d_C);
    CHECK_ERROR(cudaMemcpy(gpuRef.data_ptr(), d_C, nBytes, cudaMemcpyDeviceToHost));
    if (iprint)
        std::cout << gpuRef << std::endl;

    // launch the kernel setRowReadCol
    CHECK_ERROR(cudaMemset(d_C, 0, nBytes));
    setRowReadCol<<<grid, block>>>(d_C);
    CHECK_ERROR(cudaMemcpy(gpuRef.data_ptr(), d_C, nBytes, cudaMemcpyDeviceToHost));
    if (iprint)
        std::cout << gpuRef << std::endl;

    // launch the kernel setRowReadColDyn
    CHECK_ERROR(cudaMemset(d_C, 0, nBytes));
    setRowReadColDyn<<<grid, block, nBytes>>>(d_C);
    CHECK_ERROR(cudaMemcpy(gpuRef.data_ptr(), d_C, nBytes, cudaMemcpyDeviceToHost));
    if (iprint)
        std::cout << gpuRef << std::endl;

    // launch the kernel setRowReadColPad
    CHECK_ERROR(cudaMemset(d_C, 0, nBytes));
    setRowReadColPad<<<grid, block>>>(d_C);
    CHECK_ERROR(cudaMemcpy(gpuRef.data_ptr(), d_C, nBytes, cudaMemcpyDeviceToHost));
    if (iprint)
        std::cout << gpuRef << std::endl;

    // launch the kernel setRowReadColDynPad
    CHECK_ERROR(cudaMemset(d_C, 0, nBytes));
    setRowReadColDynPad<<<grid, block, (BDIMX + 1) * BDIMY * sizeof(int)>>>(d_C);
    CHECK_ERROR(cudaMemcpy(gpuRef.data_ptr(), d_C, nBytes, cudaMemcpyDeviceToHost));
    if (iprint)
        std::cout << gpuRef << std::endl;

    CHECK_ERROR(cudaFree(d_C));
    CHECK_ERROR(cudaDeviceReset());
    return EXIT_SUCCESS;
}