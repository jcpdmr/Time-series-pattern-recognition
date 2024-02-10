#include "cuda_utility.h"

int main(int argc, char **argv) {
    helloFromGPU<<<1, 100>>>();
    // destroy and clean up all resources associated with current device
    // + current process.
    cudaDeviceReset(); // CUDA functions are async...
    // the program would terminate before CUDA kernel prints
    return 0;
}