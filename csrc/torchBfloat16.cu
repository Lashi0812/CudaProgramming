#include <cuda_runtime.h>
#include <ATen/ATen.h>
#include <cuda_bf16.h>
#include <iostream>

__global__ void test_bf16(const __nv_bfloat16 *__restrict__ A,__nv_bfloat16 *B)
{
    B[threadIdx.x] = A[threadIdx.x] + A[threadIdx.x];
}

int main()
{
    auto a = at::rand({32,},at::TensorOptions().device(at::kCUDA).dtype(at::kBFloat16));
    auto b = at::zeros({32,},at::TensorOptions().device(at::kCUDA).dtype(at::kBFloat16));

    test_bf16<<<1,32>>>(reinterpret_cast<__nv_bfloat16*> (a.data_ptr()),reinterpret_cast<__nv_bfloat16*>(b.data_ptr()));
    cudaDeviceSynchronize();

    std::cout << b.allclose(a+a) << std::endl;

}

