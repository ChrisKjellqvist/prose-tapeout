//
// Created by Chris Kjellqvist on 1/4/25.
//

#ifndef TORCH_UTIL_H
#define TORCH_UTIL_H

#ifdef USE_TORCH
#include <torch/torch.h>

torch::Tensor load_tensor(const std::string &filename);

void print_small_tensor(torch::Tensor tensor);

void print_small_tensor_1d(torch::Tensor tensor);

void print_2d_tensor(torch::Tensor tensor);

void print_batched_2d_tensor(const torch::Tensor &tensor);

void print_1d_tensor(torch::Tensor tensor);

void print_1d_tensor_as_16hex(torch::Tensor tensor);

float max_pct_diff(const torch::Tensor &a, const torch::Tensor &b);

torch::Tensor random_tensor(int batch_size, int seq_len, int embed_dim);

torch::Tensor random_tensor(int M, int N);

torch::Tensor random_tensor(int dim);

torch::Tensor* multi_head_torch_tensor_to_flt_array(const torch::Tensor& tensor, int n_heads, int slice_dim);
#endif
#endif //TORCH_UTIL_H
