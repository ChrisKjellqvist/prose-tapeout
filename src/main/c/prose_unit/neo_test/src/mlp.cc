//
// Created by Christopher Kjellqvist on 10/15/24.
//
//
// Created by Entropy Xu on 9/10/24.
//

#include "prose_golden.h"
#include "torch/torch.h"
#include <iostream>
#include <torch_util.h>

#ifdef TEST_PROSE

#include "beethoven_hardware.h"
#include "float_wrapper.h"
#include "prose_impl.h"

using namespace beethoven;

#endif


int main() {
  std::string model_path = "/Users/chriskjellqvist/Code/prose/prose_rtl/c_exec/src/prose_unit/neo_test/gen_ckpts/model_ckpts";
  // weight transpose can be precomputed, so don't worry about it
  auto c_fc_weight = load_tensor(model_path + "/c_fc.weight.pt").transpose(0, 1).contiguous();
  auto c_fc_bias = load_tensor(model_path + "/c_fc.bias.pt");
  auto c_proj_weight = load_tensor(model_path + "/c_proj.weight.pt").transpose(0, 1).contiguous();
  auto c_proj_bias = load_tensor(model_path + "/c_proj.bias.pt");
  auto input = load_tensor(model_path + "/input_mlp.pt");
  auto output_golden = load_tensor(model_path + "/output_mlp.pt");

  // memory allocations
  unsigned embedding_size = c_fc_weight.size(0);
  unsigned intermediate_size = 4 * embedding_size;
  unsigned batch_size = input.size(0);
  unsigned seq_len = input.size(1);

#ifdef TEST_PROSE
  if (batch_size > 1) {
    std::cerr << "Batch size > 1 not supported" << std::endl;
    return 1;
  }
#endif

  std::cout << "embedding_size: " << embedding_size << std::endl;
  std::cout << "intermediate_size: " << intermediate_size << std::endl;
  auto c_fc_out = torch::zeros({batch_size, seq_len, intermediate_size}, torch::kFloat32);
  auto output = torch::zeros_like(output_golden);

  std::cout << "Input size: " << input.sizes() << std::endl;

  prose_isa_golden::prose_g_matmul(input.data_ptr<float>(),
                                   c_fc_weight.data_ptr<float>(),
                                   nullptr,
                                   c_fc_bias.data_ptr<float>(),
                                   c_fc_out.data_ptr<float>(),
                                   1, 2, 768, 3072);
#ifdef TEST_PROSE
  prose_float_wrapper::prose_g_matmul(input.data_ptr<float>(),
                                      c_fc_weight.data_ptr<float>(),
                                      c_fc_out.data_ptr<float>(),
                                      c_fc_bias.data_ptr<float>(),
                                      nullptr,
                                      PROSE_biasCOLS,
                                      1,
                                      2,
                                      768,
                                      3072,
                                      false);
#endif

#if !defined(TEST_PROSE) || (defined(TEST_PROSE) && defined(Prose_MCore_N))
  prose_isa_golden::prose_m_matmul(c_fc_out.data_ptr<float>(),
                       c_proj_weight.data_ptr<float>(),
                       output.data_ptr<float>(),
                       c_proj_bias.data_ptr<float>(),
                       1, 2, 3072, 768, false, nullptr);
#endif

#if defined(TEST_PROSE) && defined(Prose_MCore_N)
  prose_float_wrapper::
                  prose_m_matmul(c_fc_out.data_ptr<float>(),
                                 c_proj_weight.data_ptr<float>(),
                                 output.data_ptr<float>(),
                                 c_proj_bias.data_ptr<float>(),
                                 1,
                                 2,
                                 3072,
                                 768,
                                 false,
                                 nullptr);
#endif

  std::cout << "output_golden: " << std::endl;
  std::cout << output_golden.sizes() << std::endl;
  print_2d_tensor(output_golden[0]);
  std::cout << "output_prose: " << std::endl;
  std::cout << output.sizes() << std::endl;
  print_2d_tensor(output[0]);
  // assert all close
  auto diff = torch::abs(output - output_golden);
  std::cout << "max diff: " << diff.max().item<float>() << std::endl;
  std::cout << "max pct diff: " << ((diff / output_golden ) * 100).max().item<float>() << std::endl;
  return 0;
}