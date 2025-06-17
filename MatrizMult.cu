#include "solve.h"
#include <cuda_runtime.h>
#include <iostream>
#include <cstdio>

__global__ void matrix_multiplication_kernel(const float* A, const float* B, float* C, int M, int N, int K) {
    int col = blockIdx.x * blockDim.x + threadIdx.x; 
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (row < M && col < K){
        float sum = 0.0f;
        for (int p = 0; p < N; ++p){
            sum += A[row * N + p] * B[p * K + col];
        }
        C[row * K + col] = sum;
    }
}

// A, B, C are device pointers (i.e. pointers to memory on the GPU)
void solve(const float* A, const float* B, float* C, int M, int N, int K) {

    float *d_a = nullptr, *d_b = nullptr, *d_c = nullptr;

    cudaMalloc((void**)d_a, M * N * sizeof(float));
    cudaMalloc((void**)d_b, N * K * sizeof(float));
    cudaMalloc((void**)d_c, M * K * sizeof(float));

    cudaMemcpy(d_a, A, M * N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, B, N * K * sizeof(float), cudaMemcpyHostToDevice);

    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((K + threadsPerBlock.x - 1) / threadsPerBlock.x, (M + threadsPerBlock.y - 1) / threadsPerBlock.y);
    
    matrix_multiplication_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_b, d_c, M, N, K);
    cudaDeviceSynchronize();

    cudaMemcpy(C, d_c, M * K * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    
}

int main(){

    int N, M, K;

    std::cin>>M;
    std::cin>>N;
    std::cin>>K;

    float* A = new float[M * N];
    float* B = new float[N * K];
    float* C = new float[M * K];

    for (int i = 0; i < M; i++){
        for (int j = 0; j < N; j++){
            std::cin>>A[i * N +j ];
        }
    }

    for (int i = 0; i < N; i++){
        for (int j = 0; j < K; j++){
            std::cin>>B[i * K + j];
        }
    }

    solve(A, B, C, M, N, K);

    for (int i = 0; i < M; i++){ // Itera pelas linhas
        for (int j = 0; j < K; j++){ // Itera pelas colunas
            std::cout << C[i * K + j] << " "; // Acessa o elemento linearmente
        }
    std::cout << std::endl; 
    }

    return 0;
}
