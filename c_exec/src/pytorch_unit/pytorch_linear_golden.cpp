//
// Created by Entropy Xu on 1/9/24.
//

#include "pytorch_linear_golden.h"
#include "torch/torch.h"

using namespace torch_model;

void torch_model::linear_layer_bf16(uint16_t *input, uint16_t *weight, uint16_t *bias, uint16_t *output,
                                    unsigned input_dim, unsigned output_dim, unsigned batch_size,
                                    ActivationType activation_type) {
    // input tensor from blob and with bf16 type
    // init to random for testing purpose
    auto input_tensor = torch::randn({batch_size, input_dim}, torch::kBFloat16);
    input_tensor.contiguous();
    auto input_tensor_ptr = input_tensor.data_ptr();
    memcpy(input_tensor_ptr, input, batch_size * input_dim * sizeof(uint16_t));

    auto weight_tensor = torch::randn({output_dim, input_dim}, torch::kBFloat16);
    auto weight_tensor_ptr = weight_tensor.data_ptr();
    memcpy(weight_tensor_ptr, weight, output_dim * input_dim * sizeof(uint16_t));

    auto bias_tensor = torch::zeros({output_dim}, torch::kBFloat16);
    if (bias != nullptr) {
        auto bias_tensor_ptr = bias_tensor.data_ptr();
        memcpy(bias_tensor_ptr, bias, output_dim * sizeof(uint16_t));
    }

    /*
     * input_tensor: [batch_size, input_dim]
     * weight_tensor: [output_dim, input_dim]
     * bias_tensor: [output_dim]
     * output_tensor: [batch_size, output_dim]
     */
    auto acc = torch::nn::functional::linear(input_tensor, weight_tensor, bias_tensor);
    if (activation_type == ActivationType::GeLU) {
        acc = torch::nn::functional::gelu(acc);
    } else if (activation_type == ActivationType::Exp) {
        acc = torch::exp(acc);
    }

    acc.contiguous();

    auto data_ptr = reinterpret_cast<uint16_t *>(acc.data_ptr());
    for (int i = 0; i < batch_size * output_dim; ++i) {
        output[i] = data_ptr[i];
    }
}

void torch_model::matmul(const float *a, const float *b, float *c, unsigned M, unsigned K, unsigned N) {
    // a: M x K
    // b: K x N
    // convert a to uint16_t in bfloat16 format
    uint16_t input_float[M * K];
    for (int i = 0; i < M * K; ++i) {
        uint32_t q = reinterpret_cast<const uint32_t &>(a[i]);
        q = q >> 16;
        input_float[i] = reinterpret_cast<const uint16_t &>(q);
    }
    // convert b to uint16_t in bfloat16 format and transpose to N x K
    uint16_t weight_float[N * K];
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < K; ++j) {
            uint32_t q = reinterpret_cast<const uint32_t &>(b[j * N + i]);
            q = q >> 16;
            weight_float[i * K + j] = reinterpret_cast<const uint16_t &>(q);
        }
    }

    uint16_t output[M * N];
    for (int i = 0; i < M * N; ++i) {
        output[i] = 0;
    }

    torch_model::linear_layer_bf16(input_float, weight_float, nullptr, output,
                                   K, N, M,
                                   ActivationType::None);

    // assign back
    for (int i = 0; i < M * N; ++i) {
        uint16_t this_acc = output[i];
        uint32_t q = reinterpret_cast<const uint32_t &>(output[i]);
        q = q << 16;
        c[i] = reinterpret_cast<const float &>(q);
    }
}
