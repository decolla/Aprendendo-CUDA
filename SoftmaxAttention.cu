#include <cuda_runtime.h>
#include <cmath>
#include <algorithm>

#define BLOCK_SIZE 32


// calcula a multiplicação de matrizes Q×KT e aplica o fator de escala
__global__ void computa_attention(
    const float* Q, const float* K, float* scores,
    int M, int N, int d, float escala)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y; // Q
    int col = blockIdx.x * blockDim.x + threadIdx.x; // K

    if (row < M && col < N) {
        float toma = 0.0f;
        for (int i = 0; i < d; ++i)
            toma += Q[row * d + i] * K[col * d + i]; 
        scores[row * N + col] = toma * escala; // armazena em scores e aplica o fator de escala
    }
}

__global__ void applica_softmax(float* scores, int M, int N)
{
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= M) return; // verifica se esta processando uma linha valida

    float max = -INFINITY; // inicia max com o menor valor possivel
    for (int i = 0; i < N; ++i)
        max = fmaxf(max, scores[row * N + i]); // encontra o valor maximo na linha de scores 

    float sum = 0.0f;
    for (int i = 0; i < N; ++i) {
        scores[row * N + i] = __expf(scores[row * N + i] - max); // subtrai max e aplica exponencial
        sum += scores[row * N + i];
    }

    for (int i = 0; i < N; ++i)
        scores[row * N + i] /= sum; // normaliza os elementos dividindo por sum 
}

// calcula a multiplicacao de scores X V
__global__ void computa_output(
    const float* scores, const float* V, float* output,
    int M, int N, int d)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;  // Q/output
    int col = blockIdx.x * blockDim.x + threadIdx.x;  // d

    if (row < M && col < d) {
        float val = 0.0f;
        for (int i = 0; i < N; ++i)
            val += scores[row * N + i] * V[i * d + col]; // produto escalar de scores e V
        output[row * d + col] = val; // armazena o valor em output
    }
}

extern "C" void solve(const float* Q, const float* K, const float* V, float* output, int M, int N, int d) {
    float* scores; // mariz para armazenar o calculo de QxK
    cudaMalloc(&scores, sizeof(float) * M * N);

    dim3 block1(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid1((N + BLOCK_SIZE - 1) / BLOCK_SIZE, (M + BLOCK_SIZE - 1) / BLOCK_SIZE);

    float escala = 1.0f / sqrtf((float)d); // fator de escalonamento
    computa_attention<<<grid1, block1>>>(Q, K, scores, M, N, d, escala);

    int threads2 = 256; // trabalha em 1D
    int blocks2 = (M + threads2 - 1) / threads2;
    applica_softmax<<<blocks2, threads2>>>(scores, M, N);

    dim3 block3(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid3((d + BLOCK_SIZE - 1) / BLOCK_SIZE, (M + BLOCK_SIZE - 1) / BLOCK_SIZE);
    computa_output<<<grid3, block3>>>(scores, V, output, M, N, d);

    cudaFree(scores);
}
