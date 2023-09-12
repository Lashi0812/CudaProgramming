#include "../include/common.h"
#include <ATen/ATen.h>
#include <iostream>

#define DIM 128

__global__ void reduceGmem(int *g_idata, int *g_odata, const unsigned int N)
{
    // thread id with the block
    unsigned int tid = threadIdx.x;
    // global thread id
    unsigned int g_tid = blockDim.x * blockIdx.x + threadIdx.x;
    // move the g_idata ptr to block data ptr
    int *block_data = g_idata + blockDim.x * blockIdx.x;

    // boundary check
    if (g_tid >= N)
        return;

    // in-place reduction
    if (blockDim.x >= 1024 && tid < 512)
        block_data[tid] += block_data[tid + 512];
    __syncthreads();
    if (blockDim.x >= 512 && tid < 256)
        block_data[tid] += block_data[tid + 256];
    __syncthreads();
    if (blockDim.x >= 256 && tid < 128)
        block_data[tid] += block_data[tid + 128];
    __syncthreads();
    if (blockDim.x >= 128 && tid < 64)
        block_data[tid] += block_data[tid + 64];
    __syncthreads();

    // last few reduction will within warp , so no syncthreads needed
    if (tid < 32)
    {
        volatile int *vsmem = block_data;
        vsmem[tid] += vsmem[tid + 32];
        vsmem[tid] += vsmem[tid + 16];
        vsmem[tid] += vsmem[tid + 8];
        vsmem[tid] += vsmem[tid + 4];
        vsmem[tid] += vsmem[tid + 2];
        vsmem[tid] += vsmem[tid + 1];
    }
    // write the grid result
    if (tid == 0)
        g_odata[blockIdx.x] = block_data[0];
}

__global__ void reduceSmem(int *g_idata, int *g_odata, const unsigned int N)
{
    // static shared memory
    __shared__ int smem[DIM];

    // get block id
    unsigned int tid = threadIdx.x;

    // get the global id and use for boundary check
    unsigned int g_tid = blockDim.x * blockIdx.x + threadIdx.x;
    if (g_tid >= N)
        return;

    // convert the global pointer to block pointer
    int *block_data = g_idata + blockIdx.x * blockDim.x;

    // copy data from global memory to shared memory
    smem[tid] = block_data[tid];
    __syncthreads();

    // same old reduction  but now using the shared memory
    if (blockDim.x >= 1024 && tid < 512)
        smem[tid] += smem[tid + 512];
    __syncthreads();
    if (blockDim.x >= 512 && tid < 256)
        smem[tid] += smem[tid + 256];
    __syncthreads();
    if (blockDim.x >= 256 && tid < 128)
        smem[tid] += smem[tid + 128];
    __syncthreads();
    if (blockDim.x >= 128 && tid < 64)
        smem[tid] += smem[tid + 64];
    __syncthreads();

    // last reduce with the warp
    if (tid < 32)
    {
        volatile int *vsmem = smem;
        vsmem[tid] += vsmem[tid + 32];
        vsmem[tid] += vsmem[tid + 16];
        vsmem[tid] += vsmem[tid + 8];
        vsmem[tid] += vsmem[tid + 4];
        vsmem[tid] += vsmem[tid + 2];
        vsmem[tid] += vsmem[tid + 1];
    }
    if (tid == 0)
        g_odata[blockIdx.x] = smem[0];
}

__global__ void reduceSmemUnroll4(int *g_idata, int *g_odata, const unsigned int N)
{
    // static shared mem
    __shared__ int smem[DIM];

    // block thread id
    unsigned int tid = threadIdx.x;

    // each thread block handle four data block
    // global thread id offset
    unsigned int g_tid = 4 * blockDim.x * blockIdx.x + threadIdx.x;

    //  adding the four data block and produce single data block
    int tmp = 0;
    if (g_tid + 3 * blockDim.x < N)
    {
        int a1 = g_idata[g_tid + 0 * blockDim.x];
        int a2 = g_idata[g_tid + 1 * blockDim.x];
        int a3 = g_idata[g_tid + 2 * blockDim.x];
        int a4 = g_idata[g_tid + 3 * blockDim.x];
        tmp = a1 + a2 + a3 + a4;
    }
    smem[tid] = tmp;
    __syncthreads();

    // same old reduction in-place within block
    if (blockDim.x >= 1024 && tid < 512)
        smem[tid] += smem[tid + 512];
    __syncthreads();
    if (blockDim.x >= 512 && tid < 256)
        smem[tid] += smem[tid + 256];
    __syncthreads();
    if (blockDim.x >= 256 && tid < 128)
        smem[tid] += smem[tid + 128];
    __syncthreads();
    if (blockDim.x >= 128 && tid < 64)
        smem[tid] += smem[tid + 64];
    __syncthreads();

    // last reduce with the warp
    if (tid < 32)
    {
        volatile int *vsmem = smem;
        vsmem[tid] += vsmem[tid + 32];
        vsmem[tid] += vsmem[tid + 16];
        vsmem[tid] += vsmem[tid + 8];
        vsmem[tid] += vsmem[tid + 4];
        vsmem[tid] += vsmem[tid + 2];
        vsmem[tid] += vsmem[tid + 1];
    }
    if (tid == 0)
        g_odata[blockIdx.x] = smem[0];
}
__global__ void reduceSmemUnroll4Dyn(int *g_idata, int *g_odata, const unsigned int N)
{
    // declare dynamic shared mem
    extern __shared__ int smem[];

    // block thread id
    unsigned int tid = threadIdx.x;

    // each thread block handle four data block
    // global thread id offset
    unsigned int g_tid = 4 * blockDim.x * blockIdx.x + threadIdx.x;

    //  adding the four data block and produce single data block
    int tmp = 0;
    if (g_tid + 3 * blockDim.x < N)
    {
        int a1 = g_idata[g_tid + 0 * blockDim.x];
        int a2 = g_idata[g_tid + 1 * blockDim.x];
        int a3 = g_idata[g_tid + 2 * blockDim.x];
        int a4 = g_idata[g_tid + 3 * blockDim.x];
        tmp = a1 + a2 + a3 + a4;
    }
    smem[tid] = tmp;
    __syncthreads();

    // same old reduction in-place within block
    if (blockDim.x >= 1024 && tid < 512)
        smem[tid] += smem[tid + 512];
    __syncthreads();
    if (blockDim.x >= 512 && tid < 256)
        smem[tid] += smem[tid + 256];
    __syncthreads();
    if (blockDim.x >= 256 && tid < 128)
        smem[tid] += smem[tid + 128];
    __syncthreads();
    if (blockDim.x >= 128 && tid < 64)
        smem[tid] += smem[tid + 64];
    __syncthreads();

    // last reduce with the warp
    if (tid < 32)
    {
        volatile int *vsmem = smem;
        vsmem[tid] += vsmem[tid + 32];
        vsmem[tid] += vsmem[tid + 16];
        vsmem[tid] += vsmem[tid + 8];
        vsmem[tid] += vsmem[tid + 4];
        vsmem[tid] += vsmem[tid + 2];
        vsmem[tid] += vsmem[tid + 1];
    }
    if (tid == 0)
        g_odata[blockIdx.x] = smem[0];
}

int main()
{
    // set device
    int dev = 0;
    cudaDeviceProp devProp;
    cudaGetDeviceProperties(&devProp, dev);
    std::cout << "Using device " << dev << " : " << devProp.name << std::endl;

    // set the data size
    unsigned int nElems = 1 << 24;
    unsigned int nBytes = nElems * sizeof(int);

    // execution configuration
    dim3 block(DIM);
    dim3 grid((nElems + block.x - 1) / block.x);

    // allocate the host side
    at::Tensor h_A = at::randint(0x00, 0xFF, nElems, at::kInt);
    at::Tensor gpuRef = at::zeros(grid.x, at::kInt);

    // allocate device side
    int *d_A, *d_C;
    CHECK_ERROR(cudaMalloc((int **)&d_A, nBytes));
    CHECK_ERROR(cudaMalloc((int **)&d_C, grid.x * sizeof(int)));

    // transfer data to device
    CHECK_ERROR(cudaMemcpy(d_A, h_A.data_ptr(), nBytes, cudaMemcpyHostToDevice));
    // launch the kernel : reduceGmem
    reduceGmem<<<grid, block>>>(d_A, d_C, nElems);
    // copy back the grid result
    CHECK_ERROR(cudaMemcpy(gpuRef.data_ptr(), d_C, grid.x * sizeof(int), cudaMemcpyDeviceToHost));
    // check result
    std::cout << "Reduce using the global memory  " << (h_A.sum().allclose(gpuRef.sum()) ? "Sum Match " : "Not Match ") << gpuRef.sum() << std::endl;

    // transfer data to device
    CHECK_ERROR(cudaMemcpy(d_A, h_A.data_ptr(), nBytes, cudaMemcpyHostToDevice));
    // launch the kernel : reduceSmem
    reduceSmem<<<grid, block>>>(d_A, d_C, nElems);
    // copy back the grid result
    CHECK_ERROR(cudaMemcpy(gpuRef.data_ptr(), d_C, grid.x * sizeof(int), cudaMemcpyDeviceToHost));
    // check result
    std::cout << "Reduce using the Static Shared memory  " << (h_A.sum().allclose(gpuRef.sum()) ? "Sum Match " : "Not Match ") << gpuRef.sum() << std::endl;

    // transfer data to device
    CHECK_ERROR(cudaMemcpy(d_A, h_A.data_ptr(), nBytes, cudaMemcpyHostToDevice));
    // launch the kernel : reduceSmemUnroll4
    int unroll = 4;
    reduceSmemUnroll4<<<grid.x / unroll, block>>>(d_A, d_C, nElems);
    // copy back the grid result/
    // not changing the size of out ,just controlling the out shape using boundary
    gpuRef.zero_();
    CHECK_ERROR(cudaMemcpy(gpuRef.data_ptr(), d_C, (grid.x / unroll) * sizeof(int), cudaMemcpyDeviceToHost));
    // check result
    std::cout << "Reduce using the Static Shared memory Unroll4  " << (h_A.sum().allclose(gpuRef.sum()) ? "Sum Match " : "Not Match ") << gpuRef.sum() << std::endl;

    // transfer data to device
    CHECK_ERROR(cudaMemcpy(d_A, h_A.data_ptr(), nBytes, cudaMemcpyHostToDevice));
    // launch the kernel : reduceSmemUnroll4Dyn
    unroll = 4;
    reduceSmemUnroll4Dyn<<<grid.x / unroll, block, DIM * sizeof(int)>>>(d_A, d_C, nElems);
    // copy back the grid result/
    // not changing the size of out ,just controlling the out shape using boundary
    gpuRef.zero_();
    CHECK_ERROR(cudaMemcpy(gpuRef.data_ptr(), d_C, (grid.x / unroll) * sizeof(int), cudaMemcpyDeviceToHost));
    // check result
    std::cout << "Reduce using the DynamicShared memory Unroll4  " << (h_A.sum().allclose(gpuRef.sum()) ? "Sum Match " : "Not Match ") << gpuRef.sum() << std::endl;

    // free the gpu memory
    CHECK_ERROR(cudaFree(d_A));
    CHECK_ERROR(cudaFree(d_C));
    CHECK_ERROR(cudaDeviceReset());
    return EXIT_SUCCESS;
}