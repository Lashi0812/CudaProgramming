#include <mma.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <ATen/ATen.h>
#include <iostream>

/*
mat a ->16x16 mat b --> 16x16
                   _______
                   |@      |
                   |       |
                   |_______|
         _______    _______
        |@      |  |@      |
        |       |  |       |
        |_______|  |_______|

*/

template <typename T>
__global__ void single_wmma_kernel(T *a, T *b, float *c)
{
    // declare the fragment
    nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, 16, 16, 16, T, nvcuda::wmma::row_major> a_frag;
    nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, 16, 16, 16, T, nvcuda::wmma::row_major> b_frag;
    nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 16, 16, 16, float> c_frag;

    // init to out to zeros
    nvcuda::wmma::fill_fragment(c_frag, 0.0f);

    // load inputs
    nvcuda::wmma::load_matrix_sync(a_frag, a, 16);
    nvcuda::wmma::load_matrix_sync(b_frag, b, 16);

    // perform the mat mul
    nvcuda::wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);

    // store the output
    nvcuda::wmma::store_matrix_sync(c, c_frag, 16, nvcuda::wmma::mem_row_major);
}

/*
mat a ->16x32 mat b --> 32x16
                                 _______
                                |&      |
                                |       |
                                |_______|
                                 _______
                                |&      |
                                |       |
                                |_______|
             _______   _______   _______
            |&      | |&      | |       |
            |       | |       | |       |
            |_______| |_______| |_______|

*/
__global__ void double_wmma_kernel(half *a, half *b, float *c)
{
    // declare the fragment
    nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, 16, 16, 16, half, nvcuda::wmma::row_major> a_frag;
    nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, 16, 16, 16, half, nvcuda::wmma::row_major> b_frag;
    nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 16, 16, 16, float> c_frag;

    // init to out to zeros
    nvcuda::wmma::fill_fragment(c_frag, 0.0f);
    for (int i{0}; i < 2; i++)
    {
        // load inputs
        nvcuda::wmma::load_matrix_sync(a_frag, &a[16 * i], 32);
        nvcuda::wmma::load_matrix_sync(b_frag, &b[256 * i], 16);

        // perform the mat mul
        nvcuda::wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
    }

    // store the output
    nvcuda::wmma::store_matrix_sync(c, c_frag, 16, nvcuda::wmma::mem_row_major);
}
/*
a -> 32X32  B-->32X16
                               _______ 
                              |&      |
                              |       |
                              |_______|
                               _______ 
                              |&      |
                              |       |
                              |_______|
             _______  _______  _______ 
            |&      ||&      ||       |
            |       ||       ||       |
            |_______||_______||_______|
             _______  _______  _______ 
            |&      ||&      ||       |
            |       ||       ||       |
            |_______||_______||_______|

*/

__global__ void wmma_without_loop(half *a, half *b, float *c)
{
    // declare the fragment
    nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, 16, 16, 16, half, nvcuda::wmma::row_major> a_frag;
    nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, 16, 16, 16, half, nvcuda::wmma::row_major> b_frag;
    nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 16, 16, 16, float> c_frag;

    // init to out to zeros
    nvcuda::wmma::fill_fragment(c_frag, 0.0f);
    
    // load inputs
    nvcuda::wmma::load_matrix_sync(a_frag, &a[0], 32);
    nvcuda::wmma::load_matrix_sync(b_frag, &b[0], 16);

    // perform the mat mul
    nvcuda::wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);

    // load inputs
    nvcuda::wmma::load_matrix_sync(a_frag, &a[16], 32);
    nvcuda::wmma::load_matrix_sync(b_frag, &b[256], 16);

    // perform the mat mul
    nvcuda::wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);

    // store the output
    nvcuda::wmma::store_matrix_sync(&c[0], c_frag, 16, nvcuda::wmma::mem_row_major);

    nvcuda::wmma::fill_fragment(c_frag, 0.0f);
    nvcuda::wmma::load_matrix_sync(a_frag, &a[16 * 32], 32);
    nvcuda::wmma::load_matrix_sync(b_frag, &b[0], 16);

    // perform the mat mul
    nvcuda::wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);

    // load inputs
    nvcuda::wmma::load_matrix_sync(a_frag, &a[16 * 32 + 16], 32);
    nvcuda::wmma::load_matrix_sync(b_frag, &b[256], 16);

    // perform the mat mul
    nvcuda::wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
    
    // store the output
    nvcuda::wmma::store_matrix_sync(&c[256], c_frag, 16, nvcuda::wmma::mem_row_major);
}


int main()
{
    auto mat_a = at::rand({16, 16}, at::TensorOptions().device(at::kCUDA, 0).dtype(at::kHalf));
    auto mat_b = at::rand({16, 16}, at::TensorOptions().device(at::kCUDA, 0).dtype(at::kHalf));
    auto warp_out = at::zeros({16, 16}, at::TensorOptions().device(at::kCUDA, 0));
    // std::cout << mat_a.mm(mat_b) << std::endl;
    // std::cout << "Mat A " << mat_a << std::endl;
    // std::cout << "Mat B " << mat_b << std::endl;
    // std::cout << "MatMul " << mat_a.matmul(mat_b) << std::endl;
    single_wmma_kernel<half><<<1, 32>>>(reinterpret_cast<half *>(mat_a.data_ptr()),
                                        reinterpret_cast<half *>(mat_b.data_ptr()),
                                        reinterpret_cast<float *>(warp_out.data_ptr()));
    // wmma_kernel<__half><<<1,32>>>((half *)(mat_a.data_ptr<at::Half>()),
    //                             (half *)(mat_a.data_ptr<at::Half>()),
    //                             (half *)(out.data_ptr<at::Half>()));

    cudaDeviceSynchronize();
    // std::cout << warp_out.to(at::kHalf) << std::endl;
    std::cout << "single_wmma_kernel "<<((mat_a.to(at::kCPU).matmul(mat_b.to(at::kCPU)).allclose(warp_out.to(at::kCPU,at::kHalf))) ? "\u2705 All Match":"\u274C Not Match") << std::endl;

    auto mat_c = at::rand({16, 32}, at::TensorOptions().device(at::kCUDA, 0).dtype(at::kHalf));
    auto mat_d = at::rand({32, 16}, at::TensorOptions().device(at::kCUDA, 0).dtype(at::kHalf));
    auto warp_out2 = at::zeros({16, 16}, at::TensorOptions().device(at::kCUDA, 0));
    double_wmma_kernel<<<1, 32>>>(reinterpret_cast<half *>(mat_c.data_ptr()),
                                  reinterpret_cast<half *>(mat_d.data_ptr()),
                                  reinterpret_cast<float *>(warp_out2.data_ptr()));
    cudaDeviceSynchronize();
    std::cout <<"double_wmma_kernel "<<((mat_c.to(at::kCPU).matmul(mat_d.to(at::kCPU)).allclose(warp_out2.to(at::kCPU,at::kHalf))) ? "\u2705 All Match":"\u274C Not Match") << std::endl;

    auto mat_e = at::rand({32, 32}, at::TensorOptions().device(at::kCUDA, 0).dtype(at::kHalf));
    auto mat_f = at::rand({32, 16}, at::TensorOptions().device(at::kCUDA, 0).dtype(at::kHalf));
    auto warp_out3 = at::zeros({32, 16}, at::TensorOptions().device(at::kCUDA, 0));
    wmma_without_loop<<<1, 32>>>(reinterpret_cast<half *>(mat_e.data_ptr()),
                                  reinterpret_cast<half *>(mat_f.data_ptr()),
                                  reinterpret_cast<float *>(warp_out3.data_ptr()));
    cudaDeviceSynchronize();
    std::cout << "wmma_without_loop "<<((mat_e.to(at::kCPU).matmul(mat_f.to(at::kCPU)).allclose(warp_out3.to(at::kCPU,at::kHalf))) ? "\u2705 All Match":"\u274C Not Match") << std::endl;
}