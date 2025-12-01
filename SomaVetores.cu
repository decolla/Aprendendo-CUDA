#include <iostream>
#include <cuda_runtime.h>

__global__ void somaVetores (int *vetA, int *vetB, int *vetC, int N){
    for(int i = 0; i<N; i++){
        vetC[i] = vetA[i] + vetB[i];
    }
}

void SomaVetoresCuda (){

    const int N = 4;

    int h_a[N] = {1,2,3,4};
    int h_b[N] = {4,3,2,1};
    int h_c[N];

    int *d_a, *d_b, *d_c;

    cudaMalloc((void**) &d_a, N * sizeof(int));
    cudaMalloc((void**) &d_b, N * sizeof(int));
    cudaMalloc((void**) &d_c, N * sizeof(int));

    cudaMemcpy(d_a, h_a, N * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, N * sizeof(int), cudaMemcpyHostToDevice);

    int blocks = 1;
    int threads = 1;

    somaVetores <<< blocks, threads >>> (d_a, d_b, d_c, N);

    cudaDeviceSynchronize();

    cudaMemcpy(h_c, d_c, N * sizeof(int), cudaMemcpyDeviceToHost);

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    std::cout << "Soma de vetores na GPU: ";
        for (int i = 0; i < N; ++i) {
    std::cout << h_c[i] << " ";
    }
    
    std::cout << std::endl;

}

int main(){
    SomaVetoresCuda ();
    return 0;
}
