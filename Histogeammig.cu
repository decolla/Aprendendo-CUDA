#include "solve.h"
#include <cuda_runtime.h>

__global__ void histogramming (const int* input, int* histogram, int N, int num_bins) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if(idx < N){
        int bin_value = input[idx];
        if (bin_value >= 0 && bin_value < num_bins) {
            atomicAdd(&histogram[bin_value], 1);
        }
    }
}

void solve(const int* input, int* histogram, int N, int num_bins) {

    int *d_input, *d_histogram;
    
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    histogramming <<< blocksPerGrid, threadsPerBlock >>> (input, histogram, N, num_bins);
    cudaDeviceSynchronize();
}
