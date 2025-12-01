#include <cuda_runtime.h>
#include <iostream>
#include <cstdio> 
#include <vector>
#include <iomanip> 
#define TILE_SIZE 16 

// codigo correspondente aos desafios A e B
// equipe go taking 

// otimizacao do kernel
__global__ void matrix_multiplication_optimized_kernel(const float* A, const float* B, float* C, int M, int N, int K) {

    __shared__ float sA[TILE_SIZE][TILE_SIZE];
    __shared__ float sB[TILE_SIZE][TILE_SIZE];

    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y; 

    float p_value = 0.0f; 

    //mudanca feita para tamanhos maiores de entrada
    for (int tile_idx = 0; tile_idx < (N + TILE_SIZE - 1) / TILE_SIZE; ++tile_idx) {

        int a_load_row = row; 
        int a_load_col = tile_idx * TILE_SIZE + threadIdx.x; 

        int b_load_row = tile_idx * TILE_SIZE + threadIdx.y;
        int b_load_col = col; 

        if (a_load_row < M && a_load_col < N) {
            sA[threadIdx.y][threadIdx.x] = A[a_load_row * N + a_load_col];
        } else {
            sA[threadIdx.y][threadIdx.x] = 0.0f; 
        }

        if (b_load_row < N && b_load_col < K) {
            sB[threadIdx.y][threadIdx.x] = B[b_load_row * K + b_load_col];
        } else {
            sB[threadIdx.y][threadIdx.x] = 0.0f;
        }

        __syncthreads(); 

        for (int i = 0; i < TILE_SIZE; ++i) {
            p_value += sA[threadIdx.y][i] * sB[i][threadIdx.x];
        }

        __syncthreads(); 
                        
    }

    if (row < M && col < K) {
        C[row * K + col] = p_value;
    }
}


void solve(const float* A_host, const float* B_host, float* C_host, int M, int N, int K) {
    float *d_a = nullptr, *d_b = nullptr, *d_c = nullptr;


    cudaMalloc((void**)&d_a, M * N * sizeof(float)); 
    cudaMalloc((void**)&d_b, N * K * sizeof(float)); 
    cudaMalloc((void**)&d_c, M * K * sizeof(float)); 


    cudaMemcpy(d_a, A_host, M * N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, B_host, N * K * sizeof(float), cudaMemcpyHostToDevice);

    dim3 threadsPerBlock(TILE_SIZE, TILE_SIZE); 
    

    dim3 blocksPerGrid((K + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (M + threadsPerBlock.y - 1) / threadsPerBlock.y);
    

    matrix_multiplication_optimized_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_b, d_c, M, N, K);
    
    cudaDeviceSynchronize(); 

    cudaMemcpy(C_host, d_c, M * K * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
}

int main() {
    int M, N, K;

    //poderia fazer tudo N mas preferi trabalhar do jeito geral de matriz de diferentes ordens 
    std::cin >> N;
    M = N;
    K = N;

    std::vector<float> h_A(M * N);
    std::vector<float> h_B(N * K);
    std::vector<float> h_C(M * K); 

    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            std::cin >> h_A[i * N + j]; 
        }
    }

    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < K; ++j) {
            std::cin >> h_B[i * K + j]; 
        }
    }

    solve(h_A.data(), h_B.data(), h_C.data(), M, N, K);

    // esta printando com duas casas decimais 
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < K; ++j) {
            printf("%.2f ", h_C[i * K + j]); 
        }
        std::cout << std::endl;
    }

    return 0;
}
