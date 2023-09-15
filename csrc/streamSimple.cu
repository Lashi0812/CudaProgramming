#include "../include/common.h"
#include <cuda_runtime.h>
#include <iostream>
#include <iomanip>
#include <memory>

#define N 100000
#define NSTREAMS 4

__global__ void kernel_1()
{
    double sum = 0.0;
    for (int i = 0; i < N; i++)
    {
        sum = sum + tan(0.1) * tan(0.1);
    }
}

__global__ void kernel_2()
{
    double sum = 0.0;
    for (int i = 0; i < N; i++)
    {
        sum = sum + tan(0.1) * tan(0.1);
    }
}
__global__ void kernel_3()
{
    double sum = 0.0;
    for (int i = 0; i < N; i++)
    {
        sum = sum + tan(0.1) * tan(0.1);
    }
}
__global__ void kernel_4()
{
    double sum = 0.0;
    for (int i = 0; i < N; i++)
    {
        sum = sum + tan(0.1) * tan(0.1);
    }
}

int main()
{
    int n_streams = NSTREAMS;
    int isize = 1;
    int iblock = 1;

    // set up max connectioin
    char *iname = "CUDA_DEVICE_MAX_CONNECTIONS";
    setenv(iname, "32", 1);
    char *ivalue = getenv(iname);
    printf("%s = %s\n", iname, ivalue);

    // declare streams
    std::unique_ptr<cudaStream_t[]> streams(new cudaStream_t[n_streams]);

    for (int i = 0; i < n_streams; i++)
    {
        CHECK_ERROR(cudaStreamCreate(&streams[i]));
    }

    // execution configuration
    dim3 block(iblock);
    dim3 grid(isize);
    std::cout << "Launching Configuration " << grid.x << " grid of " << block.x << " block ." << std::endl;

    // create events
    cudaEvent_t start, stop;
    CHECK_ERROR(cudaEventCreate(&start));
    CHECK_ERROR(cudaEventCreate(&stop));

    // record the start event
    CHECK_ERROR(cudaEventRecord(start));

    for (int i = 0; i < n_streams; i++)
    {
        kernel_1<<<grid, block, 0, streams[i]>>>();
        kernel_2<<<grid, block, 0, streams[i]>>>();
        kernel_3<<<grid, block, 0, streams[i]>>>();
        kernel_4<<<grid, block, 0, streams[i]>>>();
    }

    // record the stop event
    CHECK_ERROR(cudaEventRecord(stop));
    CHECK_ERROR(cudaEventSynchronize(stop));
    float elapsedTime;
    CHECK_ERROR(cudaEventElapsedTime(&elapsedTime, start, stop))
    std::cout << "Measured time for parallel execution = " << std::fixed << elapsedTime / 1000.0f << "s" << std::endl;

    // release all stream
    for (int i = 0; i < n_streams; i++)
    {
        CHECK_ERROR(cudaStreamDestroy(streams[i]));
    }
    streams.reset();

    // destroy event
    CHECK_ERROR(cudaEventDestroy(start));
    CHECK_ERROR(cudaEventDestroy(stop));

    CHECK_ERROR(cudaDeviceReset());
    return EXIT_SUCCESS;
}