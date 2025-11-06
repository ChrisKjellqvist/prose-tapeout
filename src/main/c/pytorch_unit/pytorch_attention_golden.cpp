//
// Created by Entropy Xu on 1/29/24.
//

#include "pytorch_attention_golden.h"

torch::Tensor to_bf16_tensor(uint16_t* x, at::IntArrayRef shape) {
    auto tensor = torch::randn(shape, torch::kHalf);
    auto numel = tensor.numel();

    tensor.contiguous();

    auto data_ptr = tensor.data_ptr();
    memcpy(data_ptr, x, numel * sizeof(uint16_t));

    return tensor;
}

void to_uint16_array(torch::Tensor x, uint16_t* output) {
    auto numel = x.numel();
    auto data_ptr = x.data_ptr();
    memcpy(output, data_ptr, numel * sizeof(uint16_t));
}

torch::Tensor split_heads(torch::Tensor x, unsigned head_num, unsigned head_size) {
    // x: [batch_size, seq_len, dim]
    TORCH_CHECK(x.dim() == 3, "split_heads: x.dim() == 3 failed");
    auto new_shape = at::IntArrayRef({x.size(0), x.size(1), head_num, head_size});
    auto x_reshaped = x.reshape(new_shape);
    auto x_permuted = x_reshaped.permute({0, 2, 1, 3});
    // output: [batch_size, head_num, seq_len, head_size]
    return x_permuted;
}

torch::Tensor merge_heads(torch::Tensor x, unsigned head_num, unsigned head_size) {
    // x: [batch_size, head_num, seq_len, head_size]
    TORCH_CHECK(x.dim() == 4, "merge_heads: x.dim() == 4 failed");
    x = x.permute({0, 2, 1, 3});
    std::cout << x.sizes() << std::endl;
    auto new_shape = at::IntArrayRef({x.size(0), x.size(1), head_num * head_size});
    auto x_reshaped = x.reshape(new_shape);
    // output: [batch_size, seq_len, dim]
    return x_reshaped;
}

torch::Tensor attention(torch::Tensor query, torch::Tensor key, torch::Tensor value) {
    // query: [batch_size, head_num, seq_len, head_size]
    // key: [batch_size, head_num, seq_len, head_size]
    // value: [batch_size, head_num, seq_len, head_size]

    // convert to fp32 for bmm
    query = query.to(torch::kFloat32);
    key = key.to(torch::kFloat32);
    value = value.to(torch::kFloat32);

    auto attn_weights = torch::matmul(query, key.transpose(-1, -2));
    auto attn_weights_softmax = torch::softmax(attn_weights, -1);
    auto attn_output = torch::matmul(attn_weights_softmax, value);
    attn_output = attn_output.to(torch::kBFloat16);

    // attn_output: [batch_size, head_num, seq_len, head_size]
    return attn_output;
}

/*
 * K: [batch_size, seq_len, dim]
 * Q: [batch_size, seq_len, dim]
 * V: [batch_size, seq_len, dim]
 * output: [batch_size, seq_len, dim]
 * dim = head_num * head_size
 * scalar: 1 / sqrt(head_size)
 */
void torch_model::multihead_attention_bf16(uint16_t *K, uint16_t *Q, uint16_t *V, uint16_t *output,
                                           unsigned batch_size, unsigned seq_len, unsigned head_num, unsigned head_size,
                                           float scalar) {
    auto dim = head_num * head_size;
    at::IntArrayRef k_size = {batch_size, seq_len, dim};
    at::IntArrayRef q_size = {batch_size, seq_len, dim};
    at::IntArrayRef v_size = {batch_size, seq_len, dim};
    auto k_tensor = to_bf16_tensor(K, k_size);
    auto q_tensor = to_bf16_tensor(Q, q_size);
    auto v_tensor = to_bf16_tensor(V, v_size);

    // split into heads
    auto key = split_heads(k_tensor, head_num, head_size);
    auto query = split_heads(q_tensor, head_num, head_size);
    auto value = split_heads(v_tensor, head_num, head_size);

    // call bmm from pytorch
    auto attn_output = attention(query, key, value);

    // merge heads
    auto output_tensor = merge_heads(attn_output, head_num, head_size);

    // copy back to output
    to_uint16_array(output_tensor, output);
}
