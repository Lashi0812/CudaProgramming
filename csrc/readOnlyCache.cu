#include <ATen/ATen.h>
#include <iostream>

__global__ void withoutReadCache(float *in ,float * out)
{
    int idx = threadIdx.y * blockDim.x + threadIdx.x;
    out[idx] = in[idx];
}

__global__ void withReadCacheHint(float *__restrict__ in ,float * out)
{
    int idx = threadIdx.y * blockDim.x + threadIdx.x;
    out[idx] = in[idx];
}

__global__ void withReadCacheFunc(float *__restrict__ in ,float * out)
{
    int idx = threadIdx.y * blockDim.x + threadIdx.x;
    out[idx] = __ldg(&in[idx]);
}

int main()
{
    int inner = 32, outer = 32;
    int nElems = inner * outer;
    int nBytes = nElems * sizeof(float);

    at::Tensor h_mat = at::rand({outer, inner});
    at::Tensor h_out = at::zeros_like(h_mat);

    float *d_mat, *d_out;
    cudaMalloc((void **)&d_mat, nBytes);
    cudaMalloc((void **)&d_out, nBytes);

    cudaMemcpy(d_mat, h_mat.data_ptr(), nBytes, cudaMemcpyHostToDevice);

    dim3 block(inner, outer);
    dim3 grid((inner + block.x - 1) / block.x, (outer + block.y - 1) / block.y);

    withoutReadCache<<<grid,block>>>(d_mat,d_out);
    withReadCacheHint<<<grid,block>>>(d_mat,d_out);
    withReadCacheFunc<<<grid,block>>>(d_mat,d_out);

    cudaFree(d_mat);
    cudaFree(d_out);
    cudaDeviceReset();
}