#include "solve.h"
#include <cuda_runtime.h>
#include <stdio.h>

__global__ void block_reduce(const float* input, float* output, int N) {
    extern __shared__ float sdata[];
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x * 2 + threadIdx.x;

    float sum = 0.0f;
    if (idx < N) sum += input[idx];
    if (idx + blockDim.x < N) sum += input[idx + blockDim.x];
    sdata[tid] = sum;
    __syncthreads();

    // Redução paralela
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s)
            sdata[tid] += sdata[tid + s];
        __syncthreads();
    }

    if (tid == 0)
        output[blockIdx.x] = sdata[0];
}

void solve(const float* input, float* output, int N) {
    float *d_input, *d_in, *d_out;
    int TPB = 256;
    int blocks = (N + TPB * 2 - 1) / (TPB * 2);

    cudaMalloc(&d_input, N * sizeof(float));
    cudaMemcpy(d_input, input, N * sizeof(float), cudaMemcpyHostToDevice);

    cudaMalloc(&d_in, blocks * sizeof(float));
    cudaMalloc(&d_out, blocks * sizeof(float));

    // Primeira redução
    block_reduce<<<blocks, TPB, TPB * sizeof(float)>>>(d_input, d_in, N);
    cudaDeviceSynchronize();

    int current_n = blocks;
    while (current_n > 1) {
        int next_blocks = (current_n + TPB * 2 - 1) / (TPB * 2);
        block_reduce<<<next_blocks, TPB, TPB * sizeof(float)>>>(d_in, d_out, current_n);
        cudaDeviceSynchronize();

        // Swap buffers
        float* temp = d_in;
        d_in = d_out;
        d_out = temp;
        current_n = next_blocks;
    }

    cudaMemcpy(output, d_in, sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_input);
    cudaFree(d_in);
    cudaFree(d_out);
}
