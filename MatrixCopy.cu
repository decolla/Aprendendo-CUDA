#include "solve.h"
#include <cuda_runtime.h>
#include <iostream>

__global__ void copy_matrix_kernel(const float* A, float* B, int N) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;

    if( x < N * N){
        B[x] = A[x];
    }
}

// A, B are device pointers (i.e. pointers to memory on the GPU)
void solve(const float* A, float* B, int N) {

    float *d_A, *d_B;

    cudaMalloc((void**)&d_A, N * N * sizeof(float));
    cudaMalloc((void**)&d_B, N * N * sizeof(float));

    cudaMemcpy(d_A, A, N * N * sizeof(float), cudaMemcpyHostToDevice);

    int total = N * N;
    int threadsPerBlock = 256;
    int blocksPerGrid = (total + threadsPerBlock - 1) / threadsPerBlock;
    copy_matrix_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, N);
    cudaDeviceSynchronize();

    cudaMemcpy(B, d_B, N * N * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_A);
    cudaFree(d_B);
} 
