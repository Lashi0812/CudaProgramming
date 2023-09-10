#include "../include/common.h"
#include <cuda_runtime.h>
#include <stdio.h>

#define MEGABYTE    (1024 * 1024)

int main(int argc, char **argv)
{
    // set up device
    int dev = 0;
    CHECK_ERROR(cudaSetDevice(dev));

    if (argc != 2) {
        printf("usage: %s <size-in-mbs>\n", argv[0]);
        return 1;
    }

    // memory size
    int n_mbs = atoi(argv[1]);
    unsigned int nbytes = n_mbs * MEGABYTE;

    // get device information
    cudaDeviceProp deviceProp;
    CHECK_ERROR(cudaGetDeviceProperties(&deviceProp, dev));

    if (!deviceProp.canMapHostMemory)
    {
        printf("Device %d does not support mapping CPU host memory!\n", dev);
        CHECK_ERROR(cudaDeviceReset());
        exit(EXIT_SUCCESS);
    }

    printf("%s starting at ", argv[0]);
    printf("device %d: %s nbyte %5.2fMB canMap %d\n", dev,
           deviceProp.name, nbytes / (1024.0f * 1024.0f),
           deviceProp.canMapHostMemory);

    // allocate pinned host memory
    float *h_a;
    double start = seconds();
    CHECK_ERROR(cudaMallocHost ((float **)&h_a, nbytes));
    double elapsed = seconds() - start;
    printf("Host memory allocation took %2.10f us\n", elapsed * 1000000.0);

    // allocate device memory
    float *d_a;
    CHECK_ERROR(cudaMalloc((float **)&d_a, nbytes));

    // initialize host memory
    memset(h_a, 0, nbytes);

    for (int i = 0; i < nbytes / sizeof(float); i++) h_a[i] = 100.10f;

    // transfer data from the host to the device
    CHECK_ERROR(cudaMemcpy(d_a, h_a, nbytes, cudaMemcpyHostToDevice));

    // transfer data from the device to the host
    CHECK_ERROR(cudaMemcpy(h_a, d_a, nbytes, cudaMemcpyDeviceToHost));

    // free memory
    CHECK_ERROR(cudaFree(d_a));
    start = seconds();
    CHECK_ERROR(cudaFreeHost(h_a));
    elapsed = seconds() - start;
    printf("Host memory deallocation took %2.10f us\n", elapsed * 1000000.0);

    // reset device
    CHECK_ERROR(cudaDeviceReset());
    return EXIT_SUCCESS;
}