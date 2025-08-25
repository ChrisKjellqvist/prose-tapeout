//
// Created by Entropy Xu on 1/29/24.
//

#include <iostream>
#include "pytorch_attention_golden.h"

int main(){
    auto batch_size = 1;
    auto seq_len = 128;
    auto head_num = 8;
    auto head_size = 128;
    auto scalar = 1.0f;
    auto dim = head_num * head_size;
    uint16_t K[batch_size * seq_len * dim];
    uint16_t Q[batch_size * seq_len * dim];
    uint16_t V[batch_size * seq_len * dim];
    uint16_t output[batch_size * seq_len * dim];

    // compiler pragma to ensure contiguous
    for (int i = 0; i < batch_size * seq_len * dim; ++i) {
        K[i] = 0x3f80;
        Q[i] = 0x3f80;
        V[i] = 0x3f80;
        output[i] = 0;
    }

    torch_model::multihead_attention_bf16(K, Q, V, output, batch_size, seq_len, head_num, head_size, scalar);

    for (int i = 0; i < batch_size * seq_len * dim; ++i) {
        std::cout << std::hex << output[i] << std::endl;
    }
    return 0;
}
