#include "../include/common.h"
#include <cuda_runtime.h>
#include <string>
#include <vector>
#include <sstream>
#include <iostream>

template <typename T>
void initialData(std::vector<T> &vec)
{
    vec.clear();
    for (decltype(vec.size()) i = 0; i < vec.size(); i++)
    {
        vec.push_back(static_cast<T>((rand() & 0xFF) / 10.0f));
    }
}

template <typename T>
bool isEqualWithEpsilon(const T &a, const T &b, const double eps = 1.0E-8)
{
    return static_cast<double>(std::abs(b - a)) <= eps;
}

template <typename T>
void sumArrayOnHost(const std::vector<T> &A, const std::vector<T> &B, std::vector<T> &C, const int offset)
{
    if (A.size() != B.size())
    {
        throw std::invalid_argument("Input vector must have same size");
    }

    C.clear();

    for (decltype(A.size()) i = offset, k = 0; i < A.size(); i++, k++)
    {
        C[k] = A[i] + B[i];
    }
}

__global__ void warmUp(const float * A, const float * B, float *C, const int offset, size_t N)
{
    size_t tid = blockDim.x * blockIdx.x + threadIdx.x;
    size_t k = tid + offset;
    if (k < N)
        C[tid] = A[k] + B[k];
}

__global__ void readOffset(const float * A, const float * B, float *C, const int offset, size_t N)
{
    size_t tid = blockDim.x * blockIdx.x + threadIdx.x;
    size_t k = tid + offset;
    if (k < N)
        C[tid] = A[k] + B[k];
}

__global__ void readOffsetUnroll2(const float * A, const float * B, float *C, const int offset, size_t N)
{
    size_t tid = blockDim.x * blockIdx.x + threadIdx.x;
    size_t k = tid + offset;
    if (k + blockDim.x < N)
    {
        C[tid] = A[k] + B[k];
        C[tid + blockDim.x] = A[k + blockDim.x] + B[k + blockDim.x];
    }
}

__global__ void readOffsetUnroll4(const float * A, const float * B, float *C, const int offset, size_t N)
{
    size_t tid = blockDim.x * blockIdx.x + threadIdx.x;
    size_t k = tid + offset;
    if (k + 3 * blockDim.x < N)
    {
        C[tid + 0 * blockDim.x] = A[k + 0 * blockDim.x] + B[k + 0 * blockDim.x];
        C[tid + 1 * blockDim.x] = A[k + 1 * blockDim.x] + B[k + 1 * blockDim.x];
        C[tid + 2 * blockDim.x] = A[k + 2 * blockDim.x] + B[k + 2 * blockDim.x];
        C[tid + 3 * blockDim.x] = A[k + 3 * blockDim.x] + B[k + 3 * blockDim.x];
    }
}

int main(int argc, char *argv[])
{
    std::vector<std::vector<int>> args;
    for (int i = 1; argv[i] != NULL; i++)
    {
        std::string input = argv[i];
        std::istringstream iss(input);
        std::vector<int> tokens;

        std::string token;
        while (std::getline(iss, token, ','))
        {
            tokens.push_back(std::stoi(token));
        }
        args.push_back(tokens);
    }

    std::vector dataSizes = args[0];
    std::vector blockSizes = args[1];
    std::vector offsets = args[2];

    for (const int &dataSize : dataSizes)
    {
        size_t nElems = 1 << dataSize;

        // allocate host side
        std::vector<float> h_A(nElems);
        std::vector<float> h_B(nElems);
        std::vector<float> hostRef(nElems);
        std::vector<float> gpuRef(nElems);

        size_t nBytes = sizeof(float) * h_A.size();

        // initial data
        initialData(h_A);
        initialData(h_B);

        // allocate on device size
        float *d_A, *d_B, *d_C;
        CHECK(cudaMalloc((void **)&d_A, nBytes));
        CHECK(cudaMalloc((void **)&d_B, nBytes));
        CHECK(cudaMalloc((void **)&d_C, nBytes));

        // copy the data from host to device
        CHECK(cudaMemcpy(d_A, h_A.data(), nBytes, cudaMemcpyHostToDevice));
        CHECK(cudaMemcpy(d_B, h_B.data(), nBytes, cudaMemcpyHostToDevice));

        for (const int &blockSize : blockSizes)
        {
            // execution configuration
            dim3 block(blockSize);
            dim3 grid((nElems + block.x - 1) / block.x);

            for (const int &offset : offsets)
            {
                sumArrayOnHost(h_A, h_B, hostRef, offset);
                gpuRef.clear();
                CHECK(cudaMemset(d_C, 0, nBytes));
                // launch the kernel : readOffset
                std::cout << "Launching readOffset <<<" << grid.x << "," << block.x << ">>> with offset " << offset << std::endl;
                readOffset<<<grid, block>>>(d_A, d_B, d_C, offset, nElems);
                CHECK(cudaDeviceSynchronize());
                

                CHECK(cudaMemcpy(&gpuRef[0], d_C, nBytes, cudaMemcpyDeviceToHost));

                bool check = std::equal(hostRef.begin(), hostRef.end(), gpuRef.begin(), gpuRef.end(),
                                        [](const double &a, const double &b)
                                        { return isEqualWithEpsilon(a, b); });

                if (!check)
                {
                    std::cout << "Array do not match for readOffset" << std::endl;
                }
                else
                {
                    std::cout << "Array do match for readOffset" << std::endl;
                }
                gpuRef.clear();
                CHECK(cudaMemset(d_C, 0, nBytes));

                // launch the kernel : readOffsetUnroll2
                int unroll = 2;
                std::cout << "Launching readOffsetUnroll2 <<<" << grid.x / unroll << "," << block.x << ">>> with offset " << offset << std::endl;
                readOffsetUnroll2<<<grid.x / unroll, block>>>(d_A, d_B, d_C, offset, nElems);
                CHECK(cudaDeviceSynchronize();)

                CHECK(cudaMemcpy(&gpuRef[0], d_C, nBytes, cudaMemcpyDeviceToHost));

                check = std::equal(hostRef.begin(), hostRef.end(), gpuRef.begin(), gpuRef.end(),
                                   [](const double &a, const double &b)
                                   { return isEqualWithEpsilon(a, b); });

                if (!check)
                {
                    std::cout << "Array do not match for readOffsetUnroll2" << std::endl;
                }
                else
                {
                    std::cout << "Array do match for readOffsetUnroll2" << std::endl;
                }
                gpuRef.clear();
                CHECK(cudaMemset(d_C, 0, nBytes));

                // launch the kernel : readOffsetUnroll4
                unroll = 4;
                std::cout << "Launching readOffsetUnroll4 <<<" << grid.x / unroll << "," << block.x << ">>> with offset " << offset << std::endl;
                readOffsetUnroll4<<<grid.x / unroll, block>>>(d_A, d_B, d_C, offset, nElems);
                CHECK(cudaDeviceSynchronize());

                CHECK(cudaMemcpy(&gpuRef[0], d_C, nBytes, cudaMemcpyDeviceToHost));

                check = std::equal(hostRef.begin(), hostRef.end(), gpuRef.begin(), gpuRef.end(),
                                   [](const double &a, const double &b)
                                   { return isEqualWithEpsilon(a, b); });

                if (!check)
                {
                    std::cout << "Array do not match for readOffsetUnroll4" << std::endl;
                }
                else
                {
                    std::cout << "Array do match for readOffsetUnroll4" << std::endl;
                }
                gpuRef.clear();
                CHECK(cudaMemset(d_C, 0, nBytes));
            }
        }
        CHECK(cudaFree(d_A));
        CHECK(cudaFree(d_B));
        CHECK(cudaFree(d_C));
    }
    CHECK(cudaDeviceReset());
    return EXIT_SUCCESS;
}