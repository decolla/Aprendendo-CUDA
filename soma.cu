#include <iostream>
#include <cuda_runtime.h>

// operação a ser realizada
__global__ void somaSimples(int *a, int *b, int *c) {
    *c = *a + *b; 
}
//como vou passar endereços de memoria em dimensões é necessario os * 
//tanto na op de soma quanto na func

void somaCuda (int h_a, int h_b){
    
    //valores a ser soamdos 
    int h_c;

    //alocar memoria na gpu
    int *d_a, *d_b, *d_c;

    //aloca o espaço de um inteiro para cada variavel a ser utilizadad na operação
    cudaMalloc((void**) &d_a, sizeof(int));
    cudaMalloc((void**) &d_b, sizeof(int));
    cudaMalloc((void**) &d_c, sizeof(int));

    //copiar dados da GPU para CPU
    cudaMemcpy(d_a, &h_a, sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, &h_b, sizeof(int), cudaMemcpyHostToDevice);
    // vai pegar o enderço do valor de h_b(origem) e armazenar no espaço alocado em d_b(destino)
    // passa o valor de armazenamento que é um int, e ordena que seja uma info da CPU para GPU

    // hora de lançar a porra do kernel
    int blocks = 1;
    int threads = 1;
    // vou usar 1 bloco e thread para a operação simples como essa
    // preciso aprender como gerenciar essas dimensoes dps 

    //lança o kenel com suas dimensoes
    somaSimples<<<blocks, threads>>>(d_a, d_b, d_c);
    
    //espera a operação ser realizada antes de mandar outro comando
    cudaDeviceSynchronize();

    // copia o valor de d_c(origem) para h_c(destino), da GPU para CPU
    cudaMemcpy(&h_c, d_c, sizeof(int), cudaMemcpyDeviceToHost);

    //libera memoria usada
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    //imprime o resultado
    std::cout << "Resultado da soma na GPU: " << h_c << std::endl;
}

int main() {
    
    int a,b;

    std::cout << "Digite dois Valores:\n";
    std::cin >> a >> b;
    std::cout<< "\n";

    //chama a função
    somaCuda(a, b);
    return 0;
}
