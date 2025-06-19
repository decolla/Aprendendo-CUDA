#include "solve.h"
#include <iostream>
#include <cuda_runtime.h>

#define BLOCK_SIZE 32

__global__ void matrix_transpose_kernel(const float* input, float* output, int rows, int cols) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < cols && y < rows){
        int out = x * rows + y;
        int inp = y * cols + x;
        output[out] = input[inp];
    }
}

// input, output are device pointers (i.e. pointers to memory on the GPU)
void solve(const float* input, float* output, int rows, int cols) {
    
    float *d_input, *d_output;

    cudaMalloc((void**)&d_input, rows * cols * sizeof(float));
    cudaMalloc((void**)&d_output, cols * rows * sizeof(float));

    cudaMemcpy(d_input, input, rows * cols * sizeof(float), cudaMemcpyHostToDevice);

    dim3 threadsPerBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 blocksPerGrid((cols + BLOCK_SIZE - 1) / BLOCK_SIZE,
                       (rows + BLOCK_SIZE - 1) / BLOCK_SIZE);

    matrix_transpose_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_input, d_output, rows, cols);
    cudaDeviceSynchronize();

    cudaMemcpy(output, d_output, cols * rows * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_input);
    cudaFree(d_output);
}

int main(){

    int rows, cols;

    std::cin>>rows;
    std::cin>>cols;

    float *input = new float[rows * cols];
    float *output = new float[cols *rows];

    for (int i = 0; i < rows; i++){
        for (int j = 0; j < cols; j++){
            std::cin >> input[i * cols + j];
        }
    }

    solve(input, output, rows, cols);

    for (int i = 0; i < cols; i++){ 
        for (int j = 0; j < rows; j++){ 
            std::cout << output[i * rows + j] << " ";
        }
    std::cout << std::endl; 
    }
}
