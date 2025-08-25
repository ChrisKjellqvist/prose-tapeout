//
// Created by Chris Kjellqvist on 1/4/25.
// The goal of this test is to ensure that we can successfully convert .pt
// files into test-compatible files so that we don't have to install pytorch
// on our M55 (not going to happen)
//

#include <iostream>
#include "torch/torch.h"

#include <interchange_format.h>
#include <prose_golden.h>
#include <torch_util.h>

int main() {
  int seq_len = 128;
  int batch_size = 2;
  int hidden_size = 768;
  int D = 768;

  // weight transpose can be precomputed, so don't worry about it
  auto c_fc_weight = torch::randn({D, hidden_size});
  auto c_fc_bias = torch::randn({hidden_size}, torch::kFloat32);
  auto input = torch::randn({batch_size, seq_len, D}, torch::kFloat32);
  auto c_fc_out = torch::zeros({batch_size, seq_len, hidden_size}, torch::kFloat32);
  prose_isa_golden::prose_g_matmul(input.data_ptr<float>(), c_fc_weight.data_ptr<float>(), nullptr,
                                   c_fc_bias.data_ptr<float>(), c_fc_out.data_ptr<float>(), batch_size, seq_len, D,
                                   hidden_size);

  // now - use interchange format to go back and forth between bf16 and make sure the epsilon is SMALL
  const interchange_format if_weight(c_fc_weight, {D, hidden_size});
  const interchange_format if_bias(c_fc_bias, {hidden_size});
  const interchange_format if_input(input, {batch_size, seq_len, D});

  if_weight.write_bf16s_to_file("weight.bf16");
  if_bias.write_bf16s_to_file("bias.bf16");
  if_input.write_bf16s_to_file("input.bf16");

  auto if_wgt2 = interchange_format::from_bf16_file("weight.bf16", {D, hidden_size});
  auto if_bias2 = interchange_format::from_bf16_file("bias.bf16", {hidden_size});
  auto if_in2 = interchange_format::from_bf16_file("input.bf16", {batch_size, seq_len, D});

  auto wgt2 = if_wgt2.get_tensor();
  auto bias2 = if_bias2.get_tensor();
  auto in2 = if_in2.get_tensor();
  auto out2 = torch::zeros({batch_size, seq_len, hidden_size}, torch::kFloat32);

  prose_isa_golden::prose_g_matmul(in2.data_ptr<float>(), wgt2.data_ptr<float>(), nullptr,
                                 bias2.data_ptr<float>(), out2.data_ptr<float>(), batch_size, seq_len, D,
                                 hidden_size);

  auto diff = max_pct_diff(out2,  c_fc_out);
  std::cout << "max_pct_diff: " << diff << "%" << std::endl;
  // seems close enough (~0.5%)
}
