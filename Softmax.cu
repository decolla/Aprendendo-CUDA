#include <cuda_runtime.h>
#include <float.h>
#include <math.h>
#include <stdlib.h>

__global__ void softmax_kernel(const float* input, float* output, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= N) {
        return;
    }

    float max_val = -INFINITY; // inicia no menor valor possivel
    for (int i=0; i<N; ++i) {
        max_val = fmaxf(max_val, input[i]); //calcula o valor max
    }

    float sum = 0.0f;
    for (int i=0; i<N; ++i) {
        sum += expf(input[i] - max_val); //realiza a soma aplicando exponencial
    }

    output[idx] = expf(input[idx] - max_val) / sum; // e^x dividindo pela soma
}


// input, output are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(const float* input, float* output, int N) {
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    softmax_kernel<<<blocksPerGrid, threadsPerBlock>>>(input, output, N);
    cudaDeviceSynchronize();
}
