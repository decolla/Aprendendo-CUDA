#include "solve.h"
#include <cuda_runtime.h>

__global__ void soma(const float* y_samples, float *soma, int n_samples) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n_samples) {
        atomicAdd(soma, y_samples[idx]);
    }
}

void solve(const float* y_samples, float *result, float a, float b, int n_samples) {
    float *d_soma;
    cudaMalloc((void**)&d_soma, sizeof(float));
    cudaMemset(d_soma, 0, sizeof(float));  // inicializa o d_soma = 0

    int threadsPerBlock = 256;
    int blocksPerGrid = (n_samples + threadsPerBlock -1)/ threadsPerBlock;
    soma<<<blocksPerGrid, threadsPerBlock>>>(y_samples, d_soma, n_samples);  // usa y_samples direto
    cudaDeviceSynchronize();

    float sum_of_samples_host;
    cudaMemcpy(&sum_of_samples_host, d_soma, sizeof(float), cudaMemcpyDeviceToHost);

    float host_result = (b - a) * (1.0f / n_samples) * sum_of_samples_host;
    cudaMemcpy(result, &host_result, sizeof(float), cudaMemcpyHostToDevice);  // envia pro device

    cudaFree(d_soma);
}
