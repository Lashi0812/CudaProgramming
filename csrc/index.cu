#include <ATen/ATen.h>
#include <iostream>

__global__ void rowMajor(const int *__restrict__ matrix, int *out, const int outerDim, const int innerDim)
{
    // treating the "thread|block.x" as the inner dim ie col
    int inner = blockDim.x * blockIdx.x + threadIdx.x;
    int outer = blockDim.y * blockIdx.y + threadIdx.y;

    int idx = outer * innerDim + inner;
    if (inner < innerDim && outer < outerDim)
        out[idx] = matrix[idx];
}

__global__ void index(int *matrix, int height, int width)
{
    // find the matrix original co-ordinate (row,col)
    int mat_row = blockIdx.x * blockDim.x + threadIdx.x;
    int mat_col = blockIdx.y * blockDim.y + threadIdx.y;

    // convert the matrix co-ordinate into linear index
    int mat_idx = mat_row * width + mat_col;

    // find the thread linear index at each block level
    int blk_th_idx = threadIdx.x * blockDim.y + threadIdx.y;
    int blk_th_row = blk_th_idx / blockDim.y;
    int blk_th_col = blk_th_idx % blockDim.y;

    // find the thread linear index at each TRANSPOSED block level
    int blk_trans_th_idx = threadIdx.y * blockDim.x + threadIdx.x;
    int blk_trans_th_row = blk_trans_th_idx / blockDim.x;
    int blk_trans_th_col = blk_trans_th_idx % blockDim.x;
    if (mat_idx == 0)
        printf("Grid Dim (%d,%d) and block Dim (%d,%d)\n", gridDim.x, gridDim.y, blockDim.x, blockDim.y);

    printf("(%d,%d) \t(%d,%d) \t(%d,%d,%d) \t(%d,%d,%d) \t(%d,%d,%d)\n", blockIdx.x, blockIdx.y, threadIdx.x, threadIdx.y, mat_row, mat_col, mat_idx, blk_th_row, blk_th_col, blk_th_idx,
           blk_trans_th_row, blk_trans_th_col, blk_trans_th_idx);
}

int main()
{
    int width = 9, height = 4;
    int nElems = height * width;
    int nBytes = nElems * sizeof(int);
    at::Tensor matrix = at::arange(nElems, at::kInt).reshape({height, width});

    // std::cout << matrix << std::endl;
    // std::cout << matrix.transpose(1, 0) << std::endl;

    // std::cout << matrix.flatten() << std::endl;
    // std::cout << matrix.transpose(1, 0).flatten() << std::endl;

    // std::cout << matrix[1][2] << std::endl;

    // device
    int *d_mat;
    cudaMalloc((int **)&d_mat, nBytes);

    cudaMemcpy(d_mat, matrix.data_ptr(), nBytes, cudaMemcpyHostToDevice);

    dim3 block(2, 3);
    dim3 grid((height + block.x - 1) / block.x, (width + block.y - 1) / block.y);

    index<<<grid, block>>>(d_mat, height, width);
    cudaDeviceSynchronize();
    cudaFree(d_mat);

    // row major order
    int outerDim = 32, innerDim = 16;
    at::Tensor mat_a = at::arange(outerDim * innerDim, at::kInt).reshape({outerDim, innerDim});
    at::Tensor mat_b = at::zeros_like(mat_a);
    nBytes = outerDim * innerDim * 4;

    dim3 block_row(innerDim, outerDim);
    dim3 grid_row((innerDim + block_row.x - 1) / block_row.x, (outerDim + block_row.y - 1) / block_row.y);

    int *d_a, *d_b;
    cudaMalloc((void **)&d_a, nBytes);
    cudaMalloc((void **)&d_b, nBytes);

    cudaMemcpy(d_a, mat_a.data_ptr(), nBytes, cudaMemcpyHostToDevice);
    rowMajor<<<grid_row, block_row>>>(d_a, d_b, outerDim, innerDim);
    cudaMemcpy(mat_b.data_ptr(), d_b, nBytes, cudaMemcpyDeviceToHost);

    std::cout << mat_a << std::endl;
    std::cout << mat_b << std::endl;

    cudaFree(d_a);
    cudaFree(d_b);

    cudaDeviceReset();
}