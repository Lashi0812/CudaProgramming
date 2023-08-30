#include <stdio.h>

__global__ void helloFromGPU()
{
    printf("Hello from the Device side %d\n",threadIdx.x);
}

int main()
{
    printf("Hello from the Host side\n");
    
    helloFromGPU<<<1,10>>>();
    // // explicit destroy and clean up all resource used by current process
    // cudaDeviceReset();

    // Explict sync bwt the host and device
    // Return the control to Host only after all the thread is executed
    cudaDeviceSynchronize();

}