//
// Created by Entropy Xu on 1/11/24.
//

#ifndef C_EXEC_PYTORCH_LINEAR_GOLDEN_H
#define C_EXEC_PYTORCH_LINEAR_GOLDEN_H

#include <iostream>
#include <cstdint>

namespace torch_model {
// define a enum type with "GeLU" and "Exp"
    enum class ActivationType {
        None,
        GeLU,
        Exp
    };

    void linear_layer_bf16(uint16_t *input, uint16_t *weight, uint16_t *bias, uint16_t *output,
                           unsigned input_dim, unsigned output_dim, unsigned batch_size,
                           ActivationType activation_type);

    void matmul(const float *a, const float *b, float *c, unsigned M, unsigned K, unsigned N);
}

#endif //C_EXEC_PYTORCH_LINEAR_GOLDEN_H
