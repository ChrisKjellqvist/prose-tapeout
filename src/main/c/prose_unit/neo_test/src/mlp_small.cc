//
// Created by Entropy Xu on 9/10/24.
//

#include "prose_golden.h"
#include "torch/torch.h"
#include <util.h>
#include <torch_util.h>
#include <iostream>

#ifdef TEST_PROSE

#include "beethoven_hardware.h"
#include "float_wrapper.h"

#endif

int main() {
  int seq_len = 4;
  int batch_size = 2;
  int hidden_size = 16;
  int D = 64;

  // weight transpose can be precomputed, so don't worry about it
  auto c_fc_weight = torch::randn({D, hidden_size});
  auto c_fc_bias = torch::randn({hidden_size}, torch::kFloat32);

  // print the tensors
  print_2d_tensor(c_fc_weight);
  std::cout << std::endl;
  print_1d_tensor(c_fc_bias);
  std::cout << std::endl;

  auto input = torch::randn({batch_size, seq_len, D}, torch::kFloat32);

  // memory allocations
  auto c_fc_out = torch::zeros({batch_size, seq_len, hidden_size}, torch::kFloat32);


  std::cout << "Input size: " << input.sizes() << std::endl;
  std::cout << "c_fc_weight: " << c_fc_weight.sizes() << std::endl;

  prose_isa_golden::prose_g_matmul(input.data_ptr<float>(),
                                   c_fc_weight.data_ptr<float>(),
                                   nullptr,
                                   c_fc_bias.data_ptr<float>(),
                                   c_fc_out.data_ptr<float>(),
                                   batch_size, seq_len, D, hidden_size);

#ifdef TEST_PROSE
  auto intermediate_tensor = torch::zeros({batch_size, seq_len, hidden_size}, torch::kFloat32);
  prose_float_wrapper::
  prose_g_matmul(input.data_ptr<float>(),
                 c_fc_weight.data_ptr<float>(),
                 intermediate_tensor.data_ptr<float>(),
                 c_fc_bias.data_ptr<float>(),
                 nullptr,
                 PROSE_biasCOLS,
                 batch_size,
                 seq_len, D, hidden_size, false);
#endif
  auto c_proj_weight = torch::randn({hidden_size, D}, torch::kFloat32);
  auto c_proj_bias = torch::randn({D}, torch::kFloat32);
  std::cout << "c_proj_weight: " << c_proj_weight.sizes() << std::endl;
  std::cout << "c_proj_bias: " << c_proj_bias.sizes() << std::endl;
  print_2d_tensor(c_proj_weight);
  std::cout << std::endl;
  auto output_tensor = torch::zeros({batch_size, seq_len, D}, torch::kFloat32);
  auto output = torch::zeros_like(output_tensor);

#if defined(TEST_PROSE)
#if defined(PROSE_MCore_N)

  prose_float_wrapper::
  prose_m_matmul(c_fc_out.data_ptr<float>(),
                 c_proj_weight.data_ptr<float>(),
                 output_tensor.data_ptr<float>(),
                 c_proj_bias.data_ptr<float>(),
                 nullptr,
                 batch_size, seq_len, hidden_size, D, true, false, false,
                 PROSE_biasCOLS);
  prose_isa_golden::prose_m_matmul(intermediate_tensor.data_ptr<float>(),
                                   c_proj_weight.data_ptr<float>(),
                                   output.data_ptr<float>(),
                                   c_proj_bias.data_ptr<float>(),
                                   batch_size, seq_len, hidden_size, D, false, nullptr);

  std::cout << "output_c++: " << std::endl;
  print_2d_tensor(output[0]);
  std::cout << "output_prose: " << std::endl;
  print_2d_tensor(output_tensor[0]);
  std::cout << "max pct diff: " << max_pct_diff(output, output_tensor) << "%" << std::endl;


#else
  // testing prose but dont want to use M-Core, just print out intermediates
  std::cout << "intermediate_tensor: " << std::endl;
  print_batched_2d_tensor(intermediate_tensor);
  std::cout << "comparison: " << std::endl;
  print_batched_2d_tensor(c_fc_out);

  std::cout << "max pct diff: " << max_pct_diff(intermediate_tensor, c_fc_out) << "%" << std::endl;

#endif
#else
  // not testing prose
  prose_isa_golden::prose_m_matmul(c_fc_out.data_ptr<float>(),
                       c_proj_weight.data_ptr<float>(),
                       output.data_ptr<float>(),
                       c_proj_bias.data_ptr<float>(),
                       1, 2, 16, 768, false, nullptr);
#endif


  //  std::cout << "output_golden: " << std::endl;
  //  std::cout << output_golden.sizes() << std::endl;
  //  print_small_tensor(output_golden[0]);
  //  std::cout << "output_c++: " << std::endl;
  //  std::cout << output.sizes() << std::endl;
  //  print_small_tensor(c_fc_out[0]);
  //  std::cout << "output_prose: " << std::endl;
  //  print_small_tensor(output_tensor[0]);


#ifdef TEST_PROSE
  beethoven::fpga_handle_t t;
  t.shutdown();
#endif
  return 0;
}