//
// Created by Entropy Xu on 1/11/24.
//

#include "pytorch_linear_golden.h"
#include <iostream>
#include <random>


void matmul(const float *a, const float *b, float *c, size_t M, size_t K, size_t N) {
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            float acc = 0;
            for (int k = 0; k < K; ++k) {
                acc += a[i * K + k] * b[k * N + j];
            }
            c[i * N + j] = acc;
        }
    }
}


int main() {
    size_t M = 2;
    size_t K = 768;
    size_t N = 768;

    std::uniform_real_distribution<float> dist(-0.1, 0.1);
    std::default_random_engine eng(1234);

    float I[M * K];
    float W[K * N];
    float O[M * N];
    float O_torch[M * N];
    // init with random
    for (int i = 0; i < M * K; ++i) {
        I[i] = dist(eng);
    }
    for (int i = 0; i < K * N; ++i) {
        W[i] = dist(eng);
    }
    for (int i = 0; i < M * N; ++i) {
        O[i] = 0;
    }
    for (int i = 0; i < M * N; ++i) {
        O_torch[i] = 0;
    }
    matmul(I, W, O, M, K, N);
    torch_model::matmul(I, W, O_torch, M, K, N);

    for (int i = 0; i < M * N; ++i) {
        uint16_t O_bf16;
        uint16_t O_torch_bf16;
        uint32_t O_hex = reinterpret_cast<uint32_t &>(O[i]);
        uint32_t O_torch_hex = reinterpret_cast<uint32_t &>(O_torch[i]);
        O_bf16 = O_hex >> 16;
        O_torch_bf16 = O_torch_hex >> 16;
        if (O_bf16 != O_torch_bf16) {
            std::cout << "Mismatch at " << i << std::endl;
            std::cout << "FP32 baseline=" << O[i] << "(0x" << std::hex << O_bf16 << ")" <<  std::endl;
            std::cout << "PyTorch BF16=" << O_torch[i] << "(0x" << std::hex << O_torch_bf16 << ")" << std::endl;
            std::cout << "Bit Error=" << std::hex << O_bf16 - O_torch_bf16 << std::endl << std::endl;
        }
    }

    return 0;
}
