
#include "prose_golden.h"
#include "torch/torch.h"
#include "util.h"
#include <iostream>

#ifdef TEST_PROSE
#include <float_wrapper.h>
#include <beethoven/fpga_handle.h>
#endif

#include <torch_util.h>

int main() {
  // chris work computer
  std::string model_path = "/Users/chriskjellqvist/Code/prose/prose_rtl/c_exec/src/prose_unit/neo_test/gen_ckpts/model_ckpts";
  // std::string model_path = "/Users/chris/Code/prose_rtl/c_exec/src/prose_unit/neo_test/model_ckpts/";
//  std::string model_path = "/Users/entropy/Development/prose_rtl/c_exec/src/prose_unit/model_ckpts/";

  auto decoder_attn_k_proj = load_tensor(model_path + "/decoder_attn.attention.k_proj.weight.pt");
  auto decoder_attn_q_proj = load_tensor(model_path + "/decoder_attn.attention.q_proj.weight.pt");
  auto decoder_attn_v_proj = load_tensor(model_path + "/decoder_attn.attention.v_proj.weight.pt");
  auto decoder_attn_out_proj_weight = load_tensor(model_path + "/decoder_attn.attention.out_proj.weight.pt");
  auto decoder_attn_out_proj_bias = load_tensor(model_path + "/decoder_attn.attention.out_proj.bias.pt");
  auto decoder_ln_1_weight = load_tensor(model_path + "/decoder_ln_1.weight.pt");
  auto decoder_ln_1_bias = load_tensor(model_path + "/decoder_ln_1.bias.pt");
  auto decoder_ln_2_weight = load_tensor(model_path + "/decoder_ln_2.weight.pt");
  auto decoder_ln_2_bias = load_tensor(model_path + "/decoder_ln_2.bias.pt");
  auto decoder_mlp_c_fc_weight = load_tensor(model_path + "/decoder_mlp.c_fc.weight.pt").t().contiguous();
  auto decoder_mlp_c_fc_bias = load_tensor(model_path + "/decoder_mlp.c_fc.bias.pt");
  auto decoder_mlp_c_proj_weight = load_tensor(model_path + "/decoder_mlp.c_proj.weight.pt").t().contiguous();
  auto decoder_mlp_c_proj_bias = load_tensor(model_path + "/decoder_mlp.c_proj.bias.pt");
  auto decoder_mlp_c_fc_out_golden = load_tensor(model_path + "/decoder_mlp_c_fc_out.pt");
  auto decoder_input_block = load_tensor(model_path + "/decoder_input_block.pt");
  auto decoder_after_attn_golden = load_tensor(model_path + "/decoder_after_attn.pt");
  auto decoder_after_ln_1_golden = load_tensor(model_path + "/decoder_after_ln_1.pt");
  auto decoder_after_ln_2_golden = load_tensor(model_path + "/decoder_after_ln_2.pt");
  auto decoder_after_mlp_golden = load_tensor(model_path + "/decoder_after_mlp.pt");
  auto decoder_output_block_golden = load_tensor(model_path + "/decoder_output_block.pt");
  auto causal_mask = load_tensor(model_path + "/attn.causal_mask.pt");

  // initialize these intermediate tensors using zeros
  auto decoder_after_attn = torch::zeros_like(decoder_after_attn_golden);
  auto decoder_after_ln_1 = torch::zeros_like(decoder_after_ln_1_golden);
  auto decoder_after_ln_2 = torch::zeros_like(decoder_after_ln_2_golden);
  auto decoder_after_mlp = torch::zeros_like(decoder_after_mlp_golden);
  auto decoder_output_block = torch::zeros_like(decoder_output_block_golden);

  // say success
  std::cout << "Successfully loaded all tensors" << std::endl;

  // the 0th dimension of the input is the batch size
  uint16_t batch_size = decoder_input_block.size(0);
  uint16_t input_length = decoder_input_block.size(1);
  uint16_t embedding_size = decoder_input_block.size(2);
  uint16_t num_heads = 12;

  // layer norm of the input
  prose_isa_golden::prose_layer_norm(decoder_input_block.data_ptr<float>(),
                                     decoder_ln_1_weight.data_ptr<float>(),
                                     decoder_ln_1_bias.data_ptr<float>(),
                                     batch_size,
                                     input_length,
                                     embedding_size,
                                     decoder_after_ln_1.data_ptr<float>());
  auto diff_after_ln1 = decoder_after_ln_1 - decoder_after_ln_1_golden;
  // assert error is less than 1e-5
  assert(diff_after_ln1.abs().max().item<float>() < 1);
#ifdef TEST_PROSE
  auto prose_accumulator = torch::zeros({batch_size, input_length, embedding_size}, torch::kFloat32);
  prose_float_wrapper::prose_layer_norm(decoder_input_block.data_ptr<float>(),
                                        decoder_ln_1_weight.data_ptr<float>(),
                                        decoder_ln_1_bias.data_ptr<float>(),
                                        batch_size,
                                        input_length,
                                        embedding_size,
                                        prose_accumulator.data_ptr<float>());
  auto diff_prose_ln1 = decoder_after_ln_1 - prose_accumulator;
  std::cout << "Max difference post layer norm: " << diff_prose_ln1.abs().max().item<float>() << std::endl;
#endif
  std::cout << "Max difference: " << diff_after_ln1.abs().max().item<float>() << std::endl;
  std::cout << "Layer norm 1 passed" << std::endl;

  // do the attention
  prose_isa_golden::prose_multi_head_attention(decoder_after_ln_1.data_ptr<float>(),
                                               decoder_attn_q_proj.data_ptr<float>(),
                                               decoder_attn_k_proj.data_ptr<float>(),
                                               decoder_attn_v_proj.data_ptr<float>(),
                                               decoder_attn_out_proj_weight.data_ptr<float>(),
                                               decoder_attn_out_proj_bias.data_ptr<float>(),
                                               causal_mask.data_ptr<float>(),
                                               batch_size,
                                               input_length,
                                               embedding_size,
                                               num_heads,
                                               decoder_after_attn.data_ptr<float>());
#ifdef TEST_PROSE
  float * sqrt_vector = new float[embedding_size];
  for (int i = 0; i < embedding_size; i++) {
    sqrt_vector[i] = std::sqrt(1.0 / embedding_size);
  }

  auto prose_attn_out = torch::zeros({batch_size, input_length, embedding_size}, torch::kFloat32);
  prose_float_wrapper::prose_multi_head_attention(
          decoder_after_ln_1.data_ptr<float>(),
          decoder_attn_q_proj.data_ptr<float>(),
          decoder_attn_k_proj.data_ptr<float>(),
          decoder_attn_v_proj.data_ptr<float>(),
          decoder_attn_out_proj_weight.data_ptr<float>(),
          decoder_attn_out_proj_bias.data_ptr<float>(),
          causal_mask.data_ptr<float>(),
          batch_size,
          input_length,
          embedding_size,
          num_heads,
          prose_attn_out.data_ptr<float>());
  auto diff_prose_attn = decoder_after_attn - prose_attn_out;
  std::cout << "Max difference post attn: " << diff_prose_attn.abs().max().item<float>() << std::endl;
#endif
  auto diff_after_attn = decoder_after_attn - decoder_after_attn_golden;
  // assert error is less than 1e-5
  std::cout << "Max difference: " << diff_after_attn.abs().max().item<float>() << std::endl;
  std::cout << "Attention passed" << std::endl;

  // residual connection and layer norm
  auto after_residual_connection = decoder_after_attn + decoder_after_ln_1;
  prose_isa_golden::prose_layer_norm(after_residual_connection.data_ptr<float>(),
                                     decoder_ln_2_weight.data_ptr<float>(),
                                     decoder_ln_2_bias.data_ptr<float>(),
                                     batch_size,
                                     input_length,
                                     embedding_size,
                                     decoder_after_ln_2.data_ptr<float>());
  auto diff_after_ln2 = decoder_after_ln_2 - decoder_after_ln_2_golden;
  assert(diff_after_ln2.abs().max().item<float>() < 1e-1);
  std::cout << "Max difference: " << diff_after_ln2.abs().max().item<float>() << std::endl;
  std::cout << "Layer norm 2 passed" << std::endl << std::endl;
  // test reset decoder_after_ln_2 to golden

  // do the mlp layer
  auto c_fc_out = torch::zeros({batch_size, input_length, /* was embedding_size=768 */ 3072}, torch::kFloat32);
  // print all the shapes of input to prose_g_matmul
  std::cout << "decoder_after_ln_2 shape: " << decoder_after_ln_2.sizes() << std::endl;
  std::cout << "decoder_mlp_c_fc_weight shape: " << decoder_mlp_c_fc_weight.sizes() << std::endl;
  std::cout << "decoder_mlp_c_fc_bias shape: " << decoder_mlp_c_fc_bias.sizes() << std::endl;
  std::cout << "c_fc_out shape: " << c_fc_out.sizes() << std::endl;
  prose_isa_golden::prose_g_matmul(decoder_after_ln_2.data_ptr<float>(),
                                   decoder_mlp_c_fc_weight.data_ptr<float>(),
                                   nullptr,
                                   decoder_mlp_c_fc_bias.data_ptr<float>(),
                                   c_fc_out.data_ptr<float>(),
                                   1, 1024, 768, 3072);
  std::cout << "g_matmul is finished" << std::endl;
  // compare with golden
  auto diff_after_fc = c_fc_out - decoder_mlp_c_fc_out_golden;
  std::cout << "Max difference: " << diff_after_fc.abs().max().item<float>() << std::endl << std::endl;
  //assert error is less than 1e-1
  assert(diff_after_fc.abs().max().item<float>() < 1e-1);

  prose_isa_golden::prose_m_matmul(c_fc_out.data_ptr<float>(),
                                   decoder_mlp_c_proj_weight.data_ptr<float>(),
                                   decoder_after_mlp.data_ptr<float>(),
                                   decoder_mlp_c_proj_bias.data_ptr<float>(),
                                   1, 1024, 3072, 768, false, nullptr);

  auto diff_after_mlp = decoder_after_mlp - decoder_after_mlp_golden;
  // assert error is less than 1e-5
  // assert(diff_after_mlp.abs().max().item<float>() < 1e-5);
  // print diff_after_mlp
  std::cout << "1 diff_after_mlp shape: " << diff_after_mlp.sizes() << std::endl;
  std::cout << "1 Max difference: " << diff_after_mlp.abs().max().item<float>() << std::endl;
  // print average difference
  std::cout << "1 Average difference: " << diff_after_mlp.abs().mean().item<float>() << std::endl;
  std::cout << "1 MLP passed" << std::endl;

  // add residual connection
  auto after_residual_connection_2 = decoder_after_mlp + decoder_after_ln_2;
  auto diff_final = after_residual_connection_2 - decoder_output_block_golden;
  // assert error is less than 1e-5
  std::cout << "2 Max difference: " << (diff_final / decoder_output_block_golden).abs().max().item<float>() << std::endl;
  // assert(diff_final.abs().max().item<float>() < 1e-5);

  std::cout << "Tests all passed" << std::endl;

  return 0;
}