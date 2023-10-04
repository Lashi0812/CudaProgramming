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
/*
    Think WARP as FUNDAMENTAL unit DON'T think in term of threads
*/
__global__ void wmma_with_loop(half *a, half *b, float *c)
{
    // out matrix warp position
    unsigned int warp_idx{(blockDim.x * blockIdx.x + threadIdx.x) / warpSize};
    unsigned int warp_col{warp_idx % 64};
    unsigned int warp_row{warp_idx / 64};
    // find the initial matrix pointer for fragment
    // warp_row * fragment * width of mat A ie (K)
    unsigned int A_ptr{warp_row * 16 * 1024};
    // warp_col * frag
    unsigned int B_ptr{warp_col * 16};

    // declare the fragment
    nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, 16, 16, 16, half, nvcuda::wmma::row_major> a_frag;
    nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, 16, 16, 16, half, nvcuda::wmma::row_major> b_frag;
    nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 16, 16, 16, float> c_frag;

    // init to out to zeros
    nvcuda::wmma::fill_fragment(c_frag, 0.0f);
    for (int i{0}; i < 64; i++)
    {
        // load inputs
        nvcuda::wmma::load_matrix_sync(a_frag, &a[A_ptr], 1024);
        nvcuda::wmma::load_matrix_sync(b_frag, &b[B_ptr], 1024);

        // perform the mat mul
        nvcuda::wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);

        // step along row for A ie next frag in row
        A_ptr += 16;
        // step along col for B ir next frag in col
        B_ptr += 16 * 1024;
    }

    // store the output
    nvcuda::wmma::store_matrix_sync(&c[(warp_row * 1024 + warp_col) * 16], c_frag, 1024, nvcuda::wmma::mem_row_major);
}
using namespace nvcuda;
__global__ void matmulT(float *C, half *A, half *B, int Ay, int Ax, int Bx)
{
    int warp = (blockDim.x * blockIdx.x + threadIdx.x) / warpSize; // warp rank in grid

    int cx = warp % (Bx / 16); // (x,y) location of active tile
    int cy = warp / (Bx / 16); // for current warp in C matrix

    int Atile_pos = cy * 16 * Bx; // start x (row) for first A tile
    int Btile_pos = cx * 16;      // start y (col) for first B tile

    // Declare the fragments as 16 x 16 tiles
    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a_frag; // A
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> b_frag; // B
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> c_frag;              // C
    wmma::fill_fragment(c_frag, 0.0f);                                        // set C = 0

    for (int k = 0; k < Ax / 16; k++)
    {                                                      // accumulate su, of row*column for C tile
        wmma::load_matrix_sync(a_frag, &A[Atile_pos], Ax); // load A as 16x16 tile
        wmma::load_matrix_sync(b_frag, &B[Btile_pos], Bx); // load B as 16x16 tile
        wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);    // C = A*B + C
        Atile_pos += 16;                                   // step along row of A
        Btile_pos += 16 * Bx;                              // step down column of B
    }
    wmma::store_matrix_sync(&C[(cy * Bx + cx) * 16], c_frag, Bx, wmma::mem_row_major);
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
    // std::cout << "single_wmma_kernel " << ((mat_a.matmul(mat_b).allclose(warp_out.to(at::kHalf))) ? "\u2705 All Match" : "\u274C Not Match") << std::endl;

    auto mat_c = at::rand({16, 32}, at::TensorOptions().device(at::kCUDA, 0).dtype(at::kHalf));
    auto mat_d = at::rand({32, 16}, at::TensorOptions().device(at::kCUDA, 0).dtype(at::kHalf));
    auto warp_out2 = at::zeros({16, 16}, at::TensorOptions().device(at::kCUDA, 0));
    double_wmma_kernel<<<1, 32>>>(reinterpret_cast<half *>(mat_c.data_ptr()),
                                  reinterpret_cast<half *>(mat_d.data_ptr()),
                                  reinterpret_cast<float *>(warp_out2.data_ptr()));
    cudaDeviceSynchronize();
    // std::cout << "double_wmma_kernel " << ((mat_c.matmul(mat_d).allclose(warp_out2.to(at::kHalf))) ? "\u2705 All Match" : "\u274C Not Match") << std::endl;

    auto mat_e = at::rand({32, 32}, at::TensorOptions().device(at::kCUDA, 0).dtype(at::kHalf));
    auto mat_f = at::rand({32, 16}, at::TensorOptions().device(at::kCUDA, 0).dtype(at::kHalf));
    auto warp_out3 = at::zeros({32, 16}, at::TensorOptions().device(at::kCUDA, 0));
    wmma_without_loop<<<1, 32>>>(reinterpret_cast<half *>(mat_e.data_ptr()),
                                 reinterpret_cast<half *>(mat_f.data_ptr()),
                                 reinterpret_cast<float *>(warp_out3.data_ptr()));
    cudaDeviceSynchronize();
    // std::cout << "wmma_without_loop " << ((mat_e.matmul(mat_f).allclose(warp_out3.to( at::kHalf))) ? "\u2705 All Match" : "\u274C Not Match") << std::endl;

    auto mat_g = at::rand({1024, 1024}, at::TensorOptions().device(at::kCUDA, 0).dtype(at::kHalf));
    auto mat_h = at::rand({1024, 1024}, at::TensorOptions().device(at::kCUDA, 0).dtype(at::kHalf));
    auto warp_out4 = at::zeros({1024, 1024}, at::TensorOptions().device(at::kCUDA, 0));
    // execution configuration
    // 1024 / 16 = 64
    // 64*64 out warps = 4096 warps
    // 4096 * 32 = 131072 tot threads
    // lets take block with 256 threads
    // then 512 grid of block need
    wmma_with_loop<<<512, 256>>>(reinterpret_cast<half *>(mat_g.data_ptr()),
                                 reinterpret_cast<half *>(mat_h.data_ptr()),
                                 reinterpret_cast<float *>(warp_out4.data_ptr()));

    // matmulT<<<512, 256>>>(reinterpret_cast<float *>(warp_out4.data_ptr()),
    //                       reinterpret_cast<half *>(mat_g.data_ptr()),
    //                        reinterpret_cast<half *>(mat_h.data_ptr()),
    //                         1024,1024,1024);
    cudaDeviceSynchronize();
    // std::cout << "wmma_with_loop " << ((mat_g.matmul(mat_h).allclose(warp_out4.to(at::kHalf))) ? "\u2705 All Match" : "\u274C Not Match") << std::endl;
    cudaDeviceReset();
}