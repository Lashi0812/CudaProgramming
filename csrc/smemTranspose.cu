#include "../include/common.h"
#include <ATen/ATen.h>
#include <iostream>

#define BINNERDIM 32
#define BOUTERDIM 16
#define PAD 2

__global__ void naiveGmem(const float *__restrict__ A, float *C, const int innerDim, const int outerDim)
{
    // using the global row ,col idx
    unsigned int ix = blockDim.x * blockIdx.x + threadIdx.x;
    unsigned int iy = blockDim.y * blockIdx.y + threadIdx.y;

    // boundary check
    if (ix < innerDim && iy < outerDim)
    {
        C[ix * outerDim + iy] = A[iy * innerDim + ix];
    }
}

__global__ void copyGmem(const float *__restrict__ A, float *C, const int innerDim, const int outerDim)
{
    // using the global row ,col idx
    unsigned int ix = blockDim.x * blockIdx.x + threadIdx.x;
    unsigned int iy = blockDim.y * blockIdx.y + threadIdx.y;

    // boundary check
    if (ix < innerDim && iy < outerDim)
    {
        C[iy * innerDim + ix] = A[iy * innerDim + ix];
    }
}

__global__ void transposeSmem(const float *__restrict__ mat, float *out, const int innerDim, const int outerDim)
{
    // static share memory
    // Storing in Row-wise so inner dim should same as block inner
    __shared__ float tile[BOUTERDIM][BINNERDIM];

    // 1. find the original matrix co-ordinate (outer,inner)
    unsigned int outer = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned int inner = blockIdx.x * blockDim.x + threadIdx.x;

    // 2. get the linear index form the co-ordinate
    unsigned int idx = outer * innerDim + inner;

    // 3.  find the thread linear index within block
    unsigned int blk_th_idx = threadIdx.y * blockDim.x + threadIdx.x;

    // 4. find the thread co-ordinate to transposed block
    // so blockDim.y will inner now
    unsigned int blk_trans_outer = blk_th_idx / blockDim.y;
    unsigned int blk_trans_inner = blk_th_idx % blockDim.y;

    // 5. Find co-ordinate for the transposed matrix
    inner = blockIdx.y * blockDim.y + blk_trans_inner;
    outer = blockIdx.x * blockDim.x + blk_trans_outer;

    unsigned int trans_idx = outer * outerDim + inner;

    if (inner < innerDim && outer < outerDim)
    {
        // read data from gmem to smem in row-wise
        tile[threadIdx.y][threadIdx.x] = mat[idx];
        __syncthreads();

        // read col-wise from smem and store in gmem
        out[trans_idx] = tile[blk_trans_inner][blk_trans_outer];
    }
}

__global__ void transposeSmemPad(const float *__restrict__ mat, float *out, const int innerDim, const int outerDim)
{
    // static share memory
    // Storing in Row-wise so inner dim should same as block inner
    __shared__ float tile[BOUTERDIM][BINNERDIM + PAD];

    // 1. find the original matrix co-ordinate (outer,inner)
    unsigned int outer = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned int inner = blockIdx.x * blockDim.x + threadIdx.x;

    // 2. get the linear index form the co-ordinate
    unsigned int idx = outer * innerDim + inner;

    // 3.  find the thread linear index within block
    unsigned int blk_th_idx = threadIdx.y * blockDim.x + threadIdx.x;

    // 4. find the thread co-ordinate to transposed block
    // so blockDim.y will inner now
    unsigned int blk_trans_outer = blk_th_idx / blockDim.y;
    unsigned int blk_trans_inner = blk_th_idx % blockDim.y;

    // 5. Find co-ordinate for the transposed matrix
    inner = blockIdx.y * blockDim.y + blk_trans_inner;
    outer = blockIdx.x * blockDim.x + blk_trans_outer;

    unsigned int trans_idx = outer * outerDim + inner;

    if (inner < innerDim && outer < outerDim)
    {
        // read data from gmem to smem in row-wise
        tile[threadIdx.y][threadIdx.x] = mat[idx];
        __syncthreads();

        // read col-wise from smem and store in gmem
        out[trans_idx] = tile[blk_trans_inner][blk_trans_outer];
    }
}

__global__ void transposeSmemUnrollPad(const float *__restrict__ mat, float *out, const int innerDim, const int outerDim)
{
    __shared__ float tile[BOUTERDIM * (BINNERDIM * 2 + PAD)];
    // each thread block will handle two data block
    // 1. find the co-ordinate of the original matrix
    unsigned int outer = blockIdx.y * 1 * blockDim.y + threadIdx.y;
    unsigned int inner = blockIdx.x * 2 * blockDim.x + threadIdx.x;

    // 2. convert into linear index
    unsigned int idx = outer * innerDim + inner;

    // 3. find the thread idx with each block
    // this will help to find the where to store the global data into shared mem
    unsigned int blk_th_idx = threadIdx.y * blockDim.x + threadIdx.x;
    // 4. get the thread co-ordinate of transposed block
    // this will help to find the which shared mem goes in global mem
    unsigned int blk_trans_outer = blk_th_idx / blockDim.y;
    unsigned int blk_trans_inner = blk_th_idx % blockDim.y;

    // 5. find the co-ordinate of transposed mat
    unsigned int trans_inner = blockIdx.y * 1 * blockDim.y + blk_trans_inner;
    unsigned int trans_outer = blockIdx.x * 2 * blockDim.x + blk_trans_outer;

    // 6. convert to linear idx for transposed matrix
    unsigned trans_idx = trans_outer * outerDim + trans_inner;

    if (inner + blockDim.x < innerDim && outer < outerDim)
    {
        // shared mem 1D convert the thread idx into linear row-major
        unsigned int row_idx = threadIdx.y * (blockDim.x * 2 + PAD) + threadIdx.x;
        tile[row_idx] = mat[idx];
        tile[row_idx + BINNERDIM] = mat[idx + BINNERDIM];

        __syncthreads();

        unsigned int col_idx = blk_trans_inner * (blockDim.x * 2 + PAD) + blk_trans_outer;
        out[trans_idx] = tile[col_idx];
        out[trans_idx + outerDim * BINNERDIM] = tile[col_idx + BINNERDIM];
    }
}

int main()
{
    // set device
    int dev = 0;
    cudaDeviceProp devProp;
    cudaGetDeviceProperties(&devProp, dev);
    std::cout << "Using device " << dev << " : " << devProp.name << std::endl;

    // 4096 * 4096 matrix
    int innerDim = 1 << 12;
    int outerDim = 1 << 12;
    size_t nBytes = innerDim * outerDim * sizeof(float);

    // execution configuration
    dim3 block(BINNERDIM, BOUTERDIM);
    dim3 grid((innerDim + block.x - 1) / block.x, (outerDim + block.y - 1) / block.y);

    // allocate the host memory
    at::Tensor h_A = at::rand({outerDim, innerDim}, at::kFloat);
    at::Tensor gpuRef = at::zeros(outerDim * innerDim, at::kFloat);

    // allocate on device side
    float *d_A, *d_C;
    CHECK_ERROR(cudaMalloc((void **)&d_A, nBytes));
    CHECK_ERROR(cudaMalloc((void **)&d_C, nBytes));

    // transfer data host to device
    CHECK_ERROR(cudaMemcpy(d_A, h_A.data_ptr(), nBytes, cudaMemcpyHostToDevice));

    // launch kernel naiveGmem
    naiveGmem<<<grid, block>>>(d_A, d_C, innerDim, outerDim);
    CHECK_ERROR(cudaMemcpy(gpuRef.data_ptr(), d_C, nBytes, cudaMemcpyDeviceToHost));
    std::cout << "Transpose using the naive global memory  " << (h_A.transpose(1, 0).allclose(gpuRef.reshape({outerDim, innerDim})) ? " Match " : "Not Match ") << std::endl;

    // launch kernel copyGmem
    copyGmem<<<grid, block>>>(d_A, d_C, innerDim, outerDim);
    CHECK_ERROR(cudaMemcpy(gpuRef.data_ptr(), d_C, nBytes, cudaMemcpyDeviceToHost));
    std::cout << "Copy using global memory  " << (h_A.allclose(gpuRef.reshape({outerDim, innerDim})) ? " Match " : "Not Match ") << std::endl;

    // launch kernel transposeSmem
    transposeSmem<<<grid, block>>>(d_A, d_C, innerDim, outerDim);
    CHECK_ERROR(cudaMemcpy(gpuRef.data_ptr(), d_C, nBytes, cudaMemcpyDeviceToHost));
    std::cout << "Transpose using the shared memory  " << (h_A.transpose(1, 0).allclose(gpuRef.reshape({outerDim, innerDim})) ? " Match " : "Not Match ") << std::endl;

    // launch kernel transposeSmemPad
    transposeSmemPad<<<grid, block>>>(d_A, d_C, innerDim, outerDim);
    CHECK_ERROR(cudaMemcpy(gpuRef.data_ptr(), d_C, nBytes, cudaMemcpyDeviceToHost));
    std::cout << "Transpose using the shared memory padded " << (h_A.transpose(1, 0).allclose(gpuRef.reshape({outerDim, innerDim})) ? " Match " : "Not Match ") << std::endl;

    // launch kernel transposeSmemUnrollPad
    dim3 grid2((innerDim + block.x * 2 - 1) / (2 * block.x), (outerDim + block.y - 1) / block.y);
    transposeSmemUnrollPad<<<grid2, block>>>(d_A, d_C, innerDim, outerDim);
    CHECK_ERROR(cudaMemcpy(gpuRef.data_ptr(), d_C, nBytes, cudaMemcpyDeviceToHost));
    std::cout << "Transpose using the shared memory padded with Unroll 2 " << (h_A.transpose(1, 0).allclose(gpuRef.reshape({outerDim, innerDim})) ? " Match " : "Not Match ") << std::endl;

    // free the device memory
    CHECK_ERROR(cudaFree(d_A));
    CHECK_ERROR(cudaFree(d_C));
    CHECK_ERROR(cudaDeviceReset());
    return EXIT_SUCCESS;
}