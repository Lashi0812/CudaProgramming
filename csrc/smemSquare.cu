#include "../include/common.h"
#include <cuda_runtime.h>
#include <ATen/ATen.h>
#include <iostream>
#include <cub/cub.cuh>

#define BDIMX 32
#define BDIMY 32
#define IPAD 1

__global__ void setRowReadRow(int *out)
{
    // static allocate shared mem
    __shared__ int tile[BDIMY][BDIMX];

    // get the global thread id
    int tid = threadIdx.y * blockDim.x + threadIdx.x;

    // store operation for shared mem
    tile[threadIdx.y][threadIdx.x] = tid;

    __syncthreads();

    // load from shared mem and store it in global
    out[tid] = tile[threadIdx.y][threadIdx.x];
}

__global__ void setRowReadRowUsingCUB(int *out)
{
    typedef cub::BlockStore<int,BDIMX,1,cub::BLOCK_STORE_DIRECT,BDIMY,1,890> BlockStore;
    __shared__ typename BlockStore::TempStorage temp_store;
    // static allocate shared mem
    __shared__ int tile[BDIMY][BDIMX];

    // get the global thread id
    int tid[1];
    tid[0] = threadIdx.y * blockDim.x + threadIdx.x;

    // store operation for shared mem
    tile[threadIdx.y][threadIdx.x] = tid[0];

    __syncthreads();
    BlockStore(temp_store).Store(out,tid);
}

__global__ void setRowReadRowWithInput(int *in,int *out)
{
    // static allocate shared mem
    __shared__ int tile[BDIMY][BDIMX];

    // get the global thread id
    int tid = threadIdx.y * blockDim.x + threadIdx.x;

    // store operation for shared mem
    tile[threadIdx.y][threadIdx.x] = in[tid] * 10;

    __syncthreads();

    // load from shared mem and store it in global
    out[tid] = tile[threadIdx.y][threadIdx.x];
}

__global__ void setRowReadRowWithInputUsingCUB(int *in,int *out)
{
    typedef cub::BlockLoad<int,BDIMX,1,cub::BLOCK_LOAD_DIRECT,BDIMY,1,890> BlockLoad;
    typedef cub::BlockStore<int,BDIMX,1,cub::BLOCK_STORE_DIRECT,BDIMY,1,890> BlockStore;

    __shared__ typename BlockLoad::TempStorage temp_load;
    __shared__ typename BlockStore::TempStorage temp_store;

    int tid[1];

    BlockLoad(temp_load).Load(in,tid); 
    tid[0] *= 10;

    __syncthreads();
    BlockStore(temp_store).Store(out,tid);
}


__global__ void setColReadCol(int *out)
{
    __shared__ int tile[BDIMX][BDIMY];

    int tid = threadIdx.y * blockDim.x + threadIdx.x;
    tile[threadIdx.x][threadIdx.y] = tid;

    __syncthreads();
    out[tid] = tile[threadIdx.x][threadIdx.y];
}

__global__ void setRowReadCol(int *out)
{
    __shared__ int tile[BDIMY][BDIMX];
    unsigned int tid = threadIdx.y * blockDim.x + threadIdx.x;
    // storing in row wise
    tile[threadIdx.y][threadIdx.x] = tid;
    __syncthreads();
    // loading in col wise
    out[tid] = tile[threadIdx.x][threadIdx.y];
}

__global__ void setRowReadColDyn(int *out)
{
    // declare the share mem variable
    // dynamic shared mem must be 1D array
    extern __shared__ int tile[];

    // get thread id along the row-wise and col-wise
    unsigned int row_idx = threadIdx.y * blockDim.x + threadIdx.x;
    unsigned int col_idx = threadIdx.x * blockDim.y + threadIdx.y;

    // store row -wise
    tile[row_idx] = row_idx;

    __syncthreads();

    // load col wise
    out[row_idx] = tile[col_idx];
}

__global__ void setRowReadColPad(int *out)
{
    // add pad to x axis
    __shared__ int tile[BDIMY][BDIMX + IPAD];

    // global tid
    unsigned int tid = threadIdx.y * blockDim.x + threadIdx.x;

    // store row-wise
    tile[threadIdx.y][threadIdx.x] = tid;

    __syncthreads();

    // load col wise
    // this there will be no bank conflict
    out[tid] = tile[threadIdx.x][threadIdx.y];
}

__global__ void setRowReadColDynPad(int *out)
{
    // declare the share mem variable
    // dynamic shared mem must be 1D array
    extern __shared__ int tile[];

    // get thread id along the row-wise and col-wise
    unsigned int row_idx = threadIdx.y * (blockDim.x + IPAD) + threadIdx.x;
    unsigned int col_idx = threadIdx.x * (blockDim.y + IPAD) + threadIdx.y;

    unsigned int gid = threadIdx.y * blockDim.x + threadIdx.x;

    // store row-wise
    tile[row_idx] = gid;

    __syncthreads();

    // load col-wise
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
    at::Tensor h_A = at::arange(nx*ny, at::kInt);
    at::Tensor gpuRef = at::zeros({nx, ny}, at::kInt);

    // device side
    int *d_A,*d_C;
    CHECK_ERROR(cudaMalloc((int **)&d_A, nBytes));
    CHECK_ERROR(cudaMalloc((int **)&d_C, nBytes));

    CHECK_ERROR(cudaMemcpy(d_A,h_A.data_ptr(),nBytes,cudaMemcpyHostToDevice));

    // execution configuration
    dim3 block(nx, ny);
    dim3 grid(1, 1);

    std::cout << "Launching with <<<(" << grid.x << "," << grid.y << "),(" << block.x << "," << block.y << ")>>>" << std::endl;

    // launch kernel : setRowReadRow
    CHECK_ERROR(cudaMemset(d_C, 0, nBytes));
    setRowReadRow<<<grid, block>>>(d_C);
    CHECK_ERROR(cudaMemcpy(gpuRef.data_ptr(), d_C, nBytes, cudaMemcpyDeviceToHost));
    if (iprint)
        std::cout << gpuRef << std::endl;
    
    // launch kernel : setRowReadRowUsingCUB
    CHECK_ERROR(cudaMemset(d_C, 0, nBytes));
    setRowReadRowUsingCUB<<<grid, block>>>(d_C);
    CHECK_ERROR(cudaMemcpy(gpuRef.data_ptr(), d_C, nBytes, cudaMemcpyDeviceToHost));
    if (iprint)
        std::cout << gpuRef << std::endl;

    // launch kernel : setRowReadRowWithInput
    CHECK_ERROR(cudaMemset(d_C, 0, nBytes));
    setRowReadRowWithInput<<<grid, block>>>(d_A,d_C);
    CHECK_ERROR(cudaMemcpy(gpuRef.data_ptr(), d_C, nBytes, cudaMemcpyDeviceToHost));
    if (iprint)
        std::cout << gpuRef << std::endl;
    
    // launch kernel : setRowReadRowWithInputUsingCUB
    CHECK_ERROR(cudaMemset(d_C, 0, nBytes));
    setRowReadRowWithInputUsingCUB<<<grid, block>>>(d_A,d_C);
    CHECK_ERROR(cudaMemcpy(gpuRef.data_ptr(), d_C, nBytes, cudaMemcpyDeviceToHost));
    if (iprint)
        std::cout << gpuRef << std::endl;

    // launch kernel : setColReadCol
    CHECK_ERROR(cudaMemset(d_C, 0, nBytes));
    setColReadCol<<<grid, block>>>(d_C);
    CHECK_ERROR(cudaMemcpy(gpuRef.data_ptr(), d_C, nBytes, cudaMemcpyDeviceToHost));
    if (iprint)
        std::cout << gpuRef << std::endl;

    // launch kernel : setRowReadCol
    CHECK_ERROR(cudaMemset(d_C, 0, nBytes));
    setRowReadCol<<<grid, block>>>(d_C);
    CHECK_ERROR(cudaMemcpy(gpuRef.data_ptr(), d_C, nBytes, cudaMemcpyDeviceToHost));
    if (iprint)
        std::cout << gpuRef << std::endl;

    // launch kernel : setRowReadColDyn
    CHECK_ERROR(cudaMemset(d_C, 0, nBytes));
    setRowReadColDyn<<<grid, block, nBytes>>>(d_C);
    CHECK_ERROR(cudaMemcpy(gpuRef.data_ptr(), d_C, nBytes, cudaMemcpyDeviceToHost));
    if (iprint)
        std::cout << gpuRef << std::endl;

    // launch kernel : setRowReadColPad
    CHECK_ERROR(cudaMemset(d_C, 0, nBytes));
    setRowReadColPad<<<grid, block>>>(d_C);
    CHECK_ERROR(cudaMemcpy(gpuRef.data_ptr(), d_C, nBytes, cudaMemcpyDeviceToHost));
    if (iprint)
        std::cout << gpuRef << std::endl;

    // launch kernel : setRowReadColDynPad
    CHECK_ERROR(cudaMemset(d_C, 0, nBytes));
    setRowReadColDynPad<<<grid, block, (BDIMX + IPAD) * BDIMY * sizeof(int)>>>(d_C);
    CHECK_ERROR(cudaMemcpy(gpuRef.data_ptr(), d_C, nBytes, cudaMemcpyDeviceToHost));
    if (iprint)
        std::cout << gpuRef << std::endl;

    CHECK_ERROR(cudaFree(d_A));
    CHECK_ERROR(cudaFree(d_C));
    CHECK_ERROR(cudaDeviceReset());
    return EXIT_SUCCESS;
}