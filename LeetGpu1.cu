#include "solve.h"
#include <cuda_runtime.h>
#include <iostream>

__global__ void vector_add(const float* A, const float* B, float* C, int N) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < N){
        C[idx] = A[idx] + B[idx];
    }
}

// A, B, C are device pointers (i.e. pointers to memory on the GPU)
void solve(const float* A, const float* B, float* C, int N) {

    //device
    float *d_a, *d_b, *d_c;

    cudaMalloc((void**)&d_a, N * sizeof(float));
    cudaMalloc((void**)&d_b, N * sizeof(float));
    cudaMalloc((void**)&d_c, N * sizeof(float));

    cudaMemcpy(d_a, A, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, B, N * sizeof(float), cudaMemcpyHostToDevice);

    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    vector_add<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_b, d_c, N);
    cudaDeviceSynchronize();

    cudaMemcpy(C, d_c, N * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_a);
    cudaFree(d_b); 
    cudaFree(d_c);

}

int main(){
     
    int N = 4;

    float A[N], B[N], C[N];

    for (int i = 0; i < N; i++){
        std::cin>>A[i];
    }
    for (int i = 0; i < N; i++){
        std::cin>>B[i];
    }

    solve(A, B, C, N);

    std::cout << "C = [";
    for (int i = 0; i < N; i++) {
        std::cout << C[i] << (i == N - 1 ? "" : ", ");
    }
    std::cout << "]" << std::endl;

    return 0;
}
