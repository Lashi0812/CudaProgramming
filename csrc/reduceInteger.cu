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

    // free memory
    free(h_idata);
    free(h_odata);

    cudaFree(d_idata);
    cudaFree(d_odata);

    cudaDeviceReset();

    return EXIT_SUCCESS;
}