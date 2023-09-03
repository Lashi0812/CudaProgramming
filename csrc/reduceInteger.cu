#include <cuda_runtime.h>
#include <stdio.h>

__global__ void reduceNeighbored(int *g_idata, int *g_odata, const unsigned int n)
{
    // thread id within the block
    unsigned int tid = threadIdx.x;
    unsigned int idx = blockDim.x * blockIdx.x + threadIdx.x;

    // convert global data pointer into local block data pointer
    // assumed 1D blocks in grid and 1D threads in block
    int *block_data = g_idata + blockDim.x * blockIdx.x;

    // boundary check
    if (idx > n)
        return;

    for (int stride = 1; stride < blockDim.x; stride *= 2)
    {
        if (tid % (2 * stride) == 0)
        {
            block_data[tid] += block_data[tid + stride];
        }
        __syncthreads();
    }

    if (tid == 0)
        g_odata[blockIdx.x] = block_data[0];
}

__global__ void reduceNeighboredLess(int *g_idata, int *g_odata, unsigned int n)
{
    unsigned int tid = threadIdx.x;
    unsigned int idx = blockDim.x * blockIdx.x + threadIdx.x;

    int *block_data = g_idata + blockDim.x * blockIdx.x;

    if (idx > n)
        return;
    for (int stride = 1; stride < blockDim.x; stride *= 2)
    {
        int index = stride * 2 * tid;
        if (index < blockDim.x)
        {
            block_data[index] += block_data[index + stride];
        }
        __syncthreads();
    }
    if (tid == 0)
        g_odata[blockIdx.x] = block_data[0];
}

__global__ void reduceInterleaved(int *g_idata, int *g_odata, unsigned int n)
{
    unsigned int tid = threadIdx.x;
    unsigned int idx = blockDim.x * blockIdx.x + threadIdx.x;

    int *block_data = g_idata + blockDim.x * blockIdx.x;

    if (idx > n)
        return;
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1)
    {
        if (tid < stride)
        {
            block_data[tid] += block_data[tid + stride];
        }
        __syncthreads();
    }
    if (tid == 0)
        g_odata[blockIdx.x] = block_data[0];
}

__global__ void reduceUnroll2(int *g_idata, int *g_odata, unsigned int n)
{
    unsigned int tid = threadIdx.x;
    unsigned int idx = blockDim.x * blockIdx.x * 2 + threadIdx.x;

    // each block can take two blocks
    int *block_data = g_idata + 2 * blockDim.x * blockIdx.x;

    // add two blocks of data using single block of threads (unroll)
    if (idx + blockDim.x < n)
        g_idata[idx] += g_idata[idx + blockDim.x];

    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1)
    {
        if (tid < stride)
        {
            block_data[tid] += block_data[tid + stride];
        }
        __syncthreads();
    }

    if (tid == 0)
        g_odata[blockIdx.x] = block_data[0];
}

__global__ void reduceUnroll4(int *g_idata, int *g_odata, unsigned int n)
{
    unsigned int tid = threadIdx.x;
    unsigned int idx = blockDim.x * blockIdx.x * 4 + threadIdx.x;

    int *block_data = g_idata + 4 * blockDim.x * blockIdx.x;

    // unroll 4
    if (idx + blockDim.x < n)
    {
        int a1 = g_idata[idx];
        int a2 = g_idata[idx + blockDim.x];
        int a3 = g_idata[idx + blockDim.x * 2];
        int a4 = g_idata[idx + blockDim.x * 3];
        g_idata[idx] = a1 + a2 + a3 + a4;
    }
    __syncthreads();

    // reduce interleaved
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1)
    {
        if (tid < stride)
        {
            block_data[tid] += block_data[tid + stride];
        }
        __syncthreads();
    }
    if (tid == 0)
        g_odata[blockIdx.x] = block_data[0];
}

__global__ void reduceUnroll8(int *g_idata, int *g_odata, unsigned int n)
{
    unsigned int tid = threadIdx.x;
    unsigned int idx = blockDim.x * blockIdx.x * 8 + threadIdx.x;

    int *block_data = g_idata + 8 * blockDim.x * blockIdx.x;

    // unroll 8
    if (idx + 7 * blockDim.x < n)
    {
        int a1 = g_idata[idx];
        int a2 = g_idata[idx + blockDim.x];
        int a3 = g_idata[idx + blockDim.x * 2];
        int a4 = g_idata[idx + blockDim.x * 3];
        int a5 = g_idata[idx + blockDim.x * 4];
        int a6 = g_idata[idx + blockDim.x * 5];
        int a7 = g_idata[idx + blockDim.x * 6];
        int a8 = g_idata[idx + blockDim.x * 7];
        g_idata[idx] = a1 + a2 + a3 + a4 + a5 + a6 + a7 + a8;
    }

    __syncthreads();

    // reduce interleaved
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1)
    {
        if (tid < stride)
        {
            block_data[tid] += block_data[tid + stride];
        }
        __syncthreads();
    }
    if (tid == 0)
        g_odata[blockIdx.x] = block_data[0];
}

int main()
{
    unsigned int n = 1 << 24;
    unsigned int nBytes = n * sizeof(int);
    printf("Array of %d size and  %d bytes\n", n, nBytes);

    // config execution
    dim3 block(512, 1);
    dim3 grid((n + block.x - 1) / block.x, 1);

    // allocate host memoey and data
    unsigned int cpu_sum = 0;
    unsigned int gpu_sum = 0;
    int *h_idata = (int *)malloc(nBytes);
    int *h_odata = (int *)malloc(grid.x * sizeof(int));

    // initailize data
    for (size_t i = 0; i < n; i++)
    {
        h_idata[i] = (int)(rand() & 0xFF);
        cpu_sum += h_idata[i];
    }

    // allocate device memory
    int *d_idata,
        *d_odata;
    cudaMalloc((void **)&d_idata, nBytes);
    cudaMalloc((void **)&d_odata, grid.x * sizeof(int));

    // copy the data host to device
    cudaMemcpy(d_idata, h_idata, nBytes, cudaMemcpyHostToDevice);

    // launch the kernel reduce Neighbor
    printf("Execution reduceNeighbored config <<<%d,%d>>>\n", grid.x, block.x);
    reduceNeighbored<<<grid, block>>>(d_idata, d_odata, n);
    cudaDeviceSynchronize();

    // get back the result
    cudaMemcpy(h_odata, d_odata, grid.x * sizeof(int), cudaMemcpyDeviceToHost);
    gpu_sum = 0;
    for (int i = 0; i < grid.x; i++)
    {
        gpu_sum += h_odata[i];
    }

    if (!(cpu_sum == gpu_sum))
        printf("Test Failed for reduce neighbor");

    // launch the kernel reduce Neighbor less
    cudaMemcpy(d_idata, h_idata, nBytes, cudaMemcpyHostToDevice);

    printf("Execution reduceNeighboredLess config <<<%d,%d>>>\n", grid.x, block.x);
    reduceNeighboredLess<<<grid, block>>>(d_idata, d_odata, n);
    cudaDeviceSynchronize();

    // get back the result
    cudaMemcpy(h_odata, d_odata, grid.x * sizeof(int), cudaMemcpyDeviceToHost);
    gpu_sum = 0;
    for (int i = 0; i < grid.x; i++)
    {
        gpu_sum += h_odata[i];
    }

    if (!(cpu_sum == gpu_sum))
        printf("Test Failed for reduce neighborLess");

    // launch the kernel reduce InterLeavced
    cudaMemcpy(d_idata, h_idata, nBytes, cudaMemcpyHostToDevice);

    printf("Execution reduceInterleaved config <<<%d,%d>>>\n", grid.x, block.x);
    reduceInterleaved<<<grid, block>>>(d_idata, d_odata, n);
    cudaDeviceSynchronize();

    // get back the result
    cudaMemcpy(h_odata, d_odata, grid.x * sizeof(int), cudaMemcpyDeviceToHost);
    gpu_sum = 0;
    for (int i = 0; i < grid.x; i++)
    {
        gpu_sum += h_odata[i];
    }

    if (!(cpu_sum == gpu_sum))
        printf("Test Failed for reduce Interleaved");

    // launch the kernel reduce unroll2
    cudaMemcpy(d_idata, h_idata, nBytes, cudaMemcpyHostToDevice);
    int unroll = 2;
    printf("Execution reduceUnroll2 config <<<%d,%d>>>\n", grid.x / unroll, block.x);
    reduceUnroll2<<<grid.x / unroll, block>>>(d_idata, d_odata, n);
    cudaDeviceSynchronize();

    // get back the result
    cudaMemcpy(h_odata, d_odata, grid.x / unroll * sizeof(int), cudaMemcpyDeviceToHost);
    gpu_sum = 0;
    for (int i = 0; i < grid.x / unroll; i++)
    {
        gpu_sum += h_odata[i];
    }

    if (!(cpu_sum == gpu_sum))
        printf("Test Failed for reduce Unroll2");

    // launch the kernel reduce unroll 4
    cudaMemcpy(d_idata, h_idata, nBytes, cudaMemcpyHostToDevice);
    unroll = 4;
    printf("Execution reduceUnroll4 config <<<%d,%d>>>\n", grid.x / unroll, block.x);
    reduceUnroll4<<<grid.x / unroll, block>>>(d_idata, d_odata, n);
    cudaDeviceSynchronize();

    // get back the result
    cudaMemcpy(h_odata, d_odata, grid.x / unroll * sizeof(int), cudaMemcpyDeviceToHost);
    gpu_sum = 0;
    for (int i = 0; i < grid.x / unroll; i++)
    {
        gpu_sum += h_odata[i];
    }

    if (!(cpu_sum == gpu_sum))
        printf("Test Failed for reduce Unroll4");

    // launch the kernel reduce unroll 8
    cudaMemcpy(d_idata, h_idata, nBytes, cudaMemcpyHostToDevice);
    unroll = 8;
    printf("Execution reduceUnroll8 config <<<%d,%d>>>\n", grid.x / unroll, block.x);
    reduceUnroll8<<<grid.x / unroll, block>>>(d_idata, d_odata, n);
    cudaDeviceSynchronize();

    // get back the result
    cudaMemcpy(h_odata, d_odata, grid.x / unroll * sizeof(int), cudaMemcpyDeviceToHost);
    gpu_sum = 0;
    for (int i = 0; i < grid.x / unroll; i++)
    {
        gpu_sum += h_odata[i];
    }

    if (!(cpu_sum == gpu_sum))
        printf("Test Failed for reduce Unroll8");

    // free memory
    free(h_idata);
    free(h_odata);

    cudaFree(d_idata);
    cudaFree(d_odata);

    cudaDeviceReset();

    return EXIT_SUCCESS;
}