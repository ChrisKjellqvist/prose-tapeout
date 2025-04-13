//
// Created by Entropy Xu on 9/10/24.
//

#include "float_wrapper.h"
#include "prose_golden.h"
#include "util.h"
#include <fstream>
#include <iostream>
#include <random>
#include "beethoven_hardware.h"
#ifdef USE_TORCH
#include <torch_util.h>
#include "torch/torch.h"
#endif



#ifdef TEST_PROSE
using namespace beethoven;
extern fpga_handle_t handle;
#endif

int main() {
  // "model_ckpts/k_proj.weight.pt"
  //    auto k_proj_w = load_tensor("model_ckpts/attn.k_proj.weight.pt");
  //    auto v_proj_w = load_tensor("model_ckpts/attn.v_proj.weight.pt");
  //    auto q_proj_w = load_tensor("model_ckpts/attn.q_proj.weight.pt");
  //    auto o_proj_w = load_tensor("model_ckpts/attn.out_proj.weight.pt");
  //    auto o_proj_b = load_tensor("model_ckpts/attn.out_proj.bias.pt");
  //    auto causal_mask = load_tensor("model_ckpts/attn.causal_mask.pt");
  //    auto zero_bias = torch::zeros_like(o_proj_b);
  //    auto golden_input = load_tensor("model_ckpts/attn.input.pt");
  //    auto golden_output = load_tensor("model_ckpts/attn.output.pt");
  // do random tensors instead of loading from file
  // make them small
  int D = 16;
  int batch_size = 2;
  int seq_len = 2;
  int num_heads = 1;
  int head_size = 8;

  std::random_device rd;
  // set seed for random number gen
  auto seed = rd();
  torch::manual_seed(3083714779L);
  std::cout << "seed: " << seed << std::endl;
  auto k_proj_w = torch::randn({num_heads, head_size, D}, torch::kFloat32);
  auto v_proj_w = torch::randn({num_heads, head_size, D}, torch::kFloat32);
  auto q_proj_w = torch::randn({num_heads, head_size, D}, torch::kFloat32);
  auto o_proj_w = torch::randn({num_heads, D, head_size}, torch::kFloat32);
  auto o_proj_b = torch::zeros({head_size}, torch::kFloat32); //torch::randn({head_size}, torch::kFloat32);
  auto causal_mask = torch::zeros({1, seq_len, seq_len}, torch::kFloat32);
//  auto causal_mask = torch::randn({1, seq_len, seq_len}, torch::kFloat32);
  auto zero_bias = torch::zeros_like(o_proj_b);
  auto input = torch::randn({batch_size, seq_len, D}, torch::kFloat32);

  k_proj_w /= 2;
  v_proj_w /= 2;
  q_proj_w /= 2;
  o_proj_w /= 2;
  o_proj_b /= 2;
  causal_mask /= 2;
  input /= 2;

  // precompute a square root vector for the softmax initialized to 1/sqrt(seq_len)
  auto sqrt_vector = torch::ones({seq_len});
  sqrt_vector /= sqrt(seq_len);
//  std::cout << "should store 1/sqrt(2)" << std::endl;
//  print_1d_tensor(sqrt_vector);
//  std::cout << std::endl;


  // 1. Check shapes
  //    assert(embed_dim == head_size * num_heads);
  // assert the shape of the causal mask to be (1, seq_len, seq_len)
  assert(causal_mask.size(0) == 1);
  assert(causal_mask.size(1) == seq_len);
  assert(causal_mask.size(2) == seq_len);

#ifdef TEST_PROSE
  auto prose_accumulator = handle.malloc(2 * batch_size * seq_len * D);
  memset(prose_accumulator.getHostAddr(), 0, 2 * batch_size * seq_len * D);
#endif

  // memory allocations
  auto attn_output = torch::zeros({batch_size, seq_len, D}, torch::kFloat32);

  std::cout << "q_proj_w_h: " << q_proj_w.sizes() << std::endl;
  std::cout << "k_proj_w_h: " << k_proj_w.sizes() << std::endl;
  std::cout << "v_proj_w_h: " << v_proj_w.sizes() << std::endl;
  std::cout << "o_proj_w_h: " << o_proj_w.sizes() << std::endl;

  // main execution flow
  for (int h = 0; h < num_heads; h++) {

    prose_isa_golden::prose_self_attention(input.data_ptr<float>(),
                                           batch_size, seq_len, D, head_size,
                                           q_proj_w[h].data_ptr<float>(),
                                           k_proj_w[h].data_ptr<float>(),
                                           v_proj_w[h].data_ptr<float>(),
                                           o_proj_w[h].data_ptr<float>(),
                                           nullptr,
                                           nullptr,
                                           nullptr,
                                           causal_mask.data_ptr<float>(),
                                           attn_output.data_ptr<float>());

    auto q_t = q_proj_w[h].transpose(-1, -2).contiguous();
    auto k_t = k_proj_w[h].transpose(-1, -2).contiguous();
    auto v_t = v_proj_w[h].transpose(-1, -2).contiguous();
    auto o_t = o_proj_w[h].transpose(-1, -2).contiguous();
#ifdef TEST_PROSE
    prose_float_wrapper::prose_self_attention(
            input.data_ptr<float>(),
            batch_size, seq_len, D, head_size,
            q_t.data_ptr<float>(),
            k_t.data_ptr<float>(),
            v_t.data_ptr<float>(),
            nullptr,
            nullptr,
            nullptr,
            causal_mask.data_ptr<float>(),
            sqrt_vector.data_ptr<float>(),
            o_t.data_ptr<float>(),
            nullptr,
            false,
            prose_accumulator);

    auto tensor_out = torch::zeros({batch_size, seq_len, D}, torch::kFloat32);
    // compare the output
    uint16_t *t = new uint16_t[batch_size * seq_len * D];
    convertPCMtoTCM((uint16_t *) prose_accumulator.getHostAddr(), t, seq_len, D, batch_size, PROSE_Nmin);
    memcpy_bf16_to_fp32(tensor_out.data_ptr<float>(), t, batch_size * seq_len * D);

    auto diff = max_pct_diff(attn_output, tensor_out);
    std::cout << "max pct diff: " << diff << "%" << std::endl;
    print_batched_2d_tensor(attn_output);
    std::cout << std::endl;
    print_batched_2d_tensor(tensor_out);

#endif
  }


  return 0;
}