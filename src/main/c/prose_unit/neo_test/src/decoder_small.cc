
#include "prose_golden.h"
#include "torch/torch.h"
#include "util.h"
#include <iostream>

#include <torch_util.h>
#ifdef TEST_PROSE
#include <beethoven_hardware.h>
#include "float_wrapper.h"

#endif

int batch_size = 2;
int D = 64;
#ifdef TEST_PROSE
int seq_len = PROSE_MCore_N * 2;
#else
int seq_len = 8;
#endif
// the 0th dimension of the input is th
// e batch size
uint16_t num_heads = 4;

int main() {
  // SEED RANDOM
  torch::manual_seed(123849);

  float max_err = 0;
  auto decoder_attn_k_proj = random_tensor(D, D);
  auto decoder_attn_q_proj = random_tensor(D, D);
  auto decoder_attn_v_proj = random_tensor(D, D);
  auto decoder_attn_out_proj_weight = random_tensor(D, D);
  auto decoder_attn_out_proj_bias = random_tensor(D);
  auto decoder_ln_1_weight = random_tensor(D);
  auto decoder_ln_1_bias = random_tensor(D);
  auto decoder_ln_2_weight = random_tensor(D);
  auto decoder_ln_2_bias = random_tensor(D);
  auto decoder_mlp_c_fc_weight = random_tensor(D, D * 3);
  auto decoder_mlp_c_fc_bias = random_tensor(D * 3);
  auto decoder_mlp_c_proj_weight = random_tensor(D * 3, D);
  auto decoder_mlp_c_proj_bias = random_tensor(D);

  auto decoder_input_block = random_tensor(batch_size, seq_len, D);
  auto causal_mask = random_tensor(seq_len, seq_len);

  // initialize these intermediate tensors using zeros
  auto decoder_after_attn = torch::zeros({batch_size, seq_len, D});
  auto prose_after_attn = torch::zeros({batch_size, seq_len, D});
  auto decoder_after_ln_1 = torch::zeros({batch_size, seq_len, D});
  auto prose_after_ln_1 = torch::zeros({batch_size, seq_len, D});
  auto decoder_after_ln_2 = torch::zeros({batch_size, seq_len, D});
  auto prose_after_ln_2 = torch::zeros({batch_size, seq_len, D});
  auto decoder_after_mlp = torch::zeros({batch_size, seq_len, D});
  auto prose_after_mlp = torch::zeros({batch_size, seq_len, D});
  auto decoder_output_block = torch::zeros({batch_size, seq_len, D});
  auto prose_output_block = torch::zeros({batch_size, seq_len, D});

  // say success
  std::cout << "Successfully loaded all tensors" << std::endl;



  // layer norm of the input
  prose_isa_golden::prose_layer_norm(decoder_input_block.data_ptr<float>(),
                                     decoder_ln_1_weight.data_ptr<float>(),
                                     decoder_ln_1_bias.data_ptr<float>(),
                                     batch_size,
                                     seq_len,
                                     D,
                                     decoder_after_ln_1.data_ptr<float>());

#ifdef TEST_PROSE
  prose_float_wrapper::prose_layer_norm(decoder_input_block.data_ptr<float>(),
                                        decoder_ln_1_weight.data_ptr<float>(),
                                        decoder_ln_1_bias.data_ptr<float>(),
                                        batch_size,
                                        seq_len,
                                        D,
                                        prose_after_ln_1.data_ptr<float>());
  auto diff = decoder_after_ln_1 - prose_after_ln_1;

  // assert error is less than 1e-5
  float pct_err = diff.abs().max().item<float>() / decoder_after_ln_1.abs().max().item<float>();
  if (pct_err > max_err) max_err = pct_err;
  if (pct_err > 0.1) {
    std::cout << "GAMMA: " << std::endl;
    print_1d_tensor(decoder_ln_1_weight);
    std::cout << "BETA: " << std::endl;
    print_1d_tensor(decoder_ln_1_bias);
    std::cout << "Input block: " << std::endl;
    print_batched_2d_tensor(decoder_input_block);
    std::cout << "Golden block: " << std::endl;
    print_batched_2d_tensor(decoder_after_ln_1);
    std::cout << "Prose block: " << std::endl;
    print_batched_2d_tensor(prose_after_ln_1);
    std::cout << "Error: " << diff.abs().max().item<float>() << std::endl;
    exit(1);
  }
    std::cout << "Layer norm 1 passed" << std::endl;
#endif

  // do the attention
  prose_isa_golden::prose_multi_head_attention(decoder_after_ln_1.data_ptr<float>(),
                                               decoder_attn_q_proj.data_ptr<float>(),
                                               decoder_attn_k_proj.data_ptr<float>(),
                                               decoder_attn_v_proj.data_ptr<float>(),
                                               decoder_attn_out_proj_weight.data_ptr<float>(),
                                               decoder_attn_out_proj_bias.data_ptr<float>(),
                                               causal_mask.data_ptr<float>(),
                                               batch_size,
                                               seq_len,
                                               D,
                                               num_heads,
                                               decoder_after_attn.data_ptr<float>());
#ifdef TEST_PROSE
  auto q_split = multi_head_torch_tensor_to_flt_array(decoder_attn_q_proj, num_heads, 0);
  auto k_split = multi_head_torch_tensor_to_flt_array(decoder_attn_k_proj, num_heads, 0);
  auto v_split = multi_head_torch_tensor_to_flt_array(decoder_attn_v_proj, num_heads, 0);
  auto o_split = multi_head_torch_tensor_to_flt_array(decoder_attn_out_proj_weight, num_heads, 1);

  float **qflt, **kflt, **vflt, **oflt;
  qflt = new float*[num_heads];
  kflt = new float*[num_heads];
  vflt = new float*[num_heads];
  oflt = new float*[num_heads];
  for (int h = 0; h < num_heads; h++) {
    qflt[h] = q_split[h].data_ptr<float>();
    kflt[h] = k_split[h].data_ptr<float>();
    vflt[h] = v_split[h].data_ptr<float>();
    oflt[h] = o_split[h].data_ptr<float>();
  }
  prose_float_wrapper::prose_multi_head_attention(decoder_after_ln_1.data_ptr<float>(),
                                                  qflt,
                                                  kflt,
                                                  vflt,
                                                  oflt,
                                                  decoder_attn_out_proj_bias.data_ptr<float>(),
                                                  causal_mask.data_ptr<float>(),
                                                  batch_size,
                                                  seq_len,
                                                  D,
                                                  num_heads,
                                                  prose_after_attn.data_ptr<float>());

  diff = decoder_after_attn - prose_after_attn;
  // assert error is less than 1e-5
  pct_err = diff.abs().max().item<float>() / decoder_after_attn.abs().max().item<float>();
  if (pct_err > max_err) max_err = pct_err;
  if (pct_err > 0.1) {
    std::cout << "Q: " << std::endl;
    print_2d_tensor(decoder_attn_q_proj);
    std::cout << "K: " << std::endl;
    print_2d_tensor(decoder_attn_k_proj);
    std::cout << "V: " << std::endl;
    print_2d_tensor(decoder_attn_v_proj);
    std::cout << "O: " << std::endl;
    print_2d_tensor(decoder_attn_out_proj_weight);
    std::cout << "B: " << std::endl;
    print_1d_tensor(decoder_attn_out_proj_bias);
    std::cout << "Input block: " << std::endl;
    print_batched_2d_tensor(decoder_after_ln_1);
    std::cout << "Golden block: " << std::endl;
    print_batched_2d_tensor(decoder_after_attn);
    std::cout << "Prose block: " << std::endl;
    print_batched_2d_tensor(prose_after_attn);
    std::cout << "Error: " << diff.abs().max().item<float>() << std::endl;
    exit(1);
  }
  std::cout << "Attention passed" << std::endl;
#endif

  // residual connection and layer norm
  auto after_residual_connection = decoder_after_attn + decoder_after_ln_1;
  prose_isa_golden::prose_layer_norm(after_residual_connection.data_ptr<float>(),
                                     decoder_ln_2_weight.data_ptr<float>(),
                                     decoder_ln_2_bias.data_ptr<float>(),
                                     batch_size,
                                     seq_len,
                                     D,
                                     decoder_after_ln_2.data_ptr<float>());

#ifdef TEST_PROSE
  auto prose_after_residual = torch::zeros({batch_size, seq_len, D});
  prose_float_wrapper::prose_matadd(decoder_after_attn.data_ptr<float>(),
                                    decoder_after_ln_1.data_ptr<float>(),
                                    prose_after_residual.data_ptr<float>(),
                                    batch_size * seq_len * D);
  prose_float_wrapper::prose_layer_norm(after_residual_connection.data_ptr<float>(),
                                        decoder_ln_2_weight.data_ptr<float>(),
                                        decoder_ln_2_bias.data_ptr<float>(),
                                        batch_size,
                                        seq_len,
                                        D,
                                        prose_after_ln_2.data_ptr<float>());
  diff = decoder_after_ln_2 - prose_after_ln_2;
  pct_err = diff.abs().max().item<float>() / decoder_after_ln_2.abs().max().item<float>();
  if (pct_err > max_err) max_err = pct_err;
  if (pct_err > 0.1) {
    std::cout << "GAMMA: " << std::endl;
    print_1d_tensor(decoder_ln_2_weight);
    std::cout << "BETA: " << std::endl;
    print_1d_tensor(decoder_ln_2_bias);
    std::cout << "Input block: " << std::endl;
    print_batched_2d_tensor(after_residual_connection);
    std::cout << "Golden block: " << std::endl;
    print_batched_2d_tensor(decoder_after_ln_2);
    std::cout << "Prose block: " << std::endl;
    print_batched_2d_tensor(prose_after_ln_2);
    std::cout << "Error: " << diff.abs().max().item<float>() << std::endl;
    exit(1);
  }
  std::cout << "Layer norm 2 passed" << std::endl;
#endif
  // do the mlp layer
  auto c_fc_out = torch::zeros({batch_size, seq_len, D * 3}, torch::kFloat32);
  prose_isa_golden::prose_g_matmul(decoder_after_ln_2.data_ptr<float>(),
                                   decoder_mlp_c_fc_weight.data_ptr<float>(),
                                   nullptr,
                                   decoder_mlp_c_fc_bias.data_ptr<float>(),
                                   c_fc_out.data_ptr<float>(),
                                   batch_size, seq_len, D, D * 3);

  prose_isa_golden::prose_m_matmul(c_fc_out.data_ptr<float>(),
                                   decoder_mlp_c_proj_weight.data_ptr<float>(),
                                   decoder_after_mlp.data_ptr<float>(),
                                   decoder_mlp_c_proj_bias.data_ptr<float>(),
                                   batch_size, seq_len, D * 3, D, false, nullptr);

#ifdef TEST_PROSE
  auto prose_c_fc_out = torch::zeros({batch_size, seq_len, D * 3}, torch::kFloat32);
  prose_float_wrapper::prose_g_matmul(decoder_after_ln_2.data_ptr<float>(),
                                      decoder_mlp_c_fc_weight.data_ptr<float>(),
                                      prose_c_fc_out.data_ptr<float>(),
                                      decoder_mlp_c_fc_bias.data_ptr<float>(),
                                      nullptr,
                                      PROSE_biasCOLS,
                                      batch_size,
                                      seq_len, D, D * 3, false);

  prose_float_wrapper::prose_m_matmul(c_fc_out.data_ptr<float>(),
                                      decoder_mlp_c_proj_weight.data_ptr<float>(),
                                      prose_after_mlp.data_ptr<float>(),
                                      decoder_mlp_c_proj_bias.data_ptr<float>(),
                                      nullptr,
                                      batch_size, seq_len, D * 3, D, true, false, false,
                                      PROSE_biasCOLS);


  diff = decoder_after_mlp - prose_after_mlp;
  pct_err = diff.abs().max().item<float>() / decoder_after_mlp.abs().max().item<float>();
  if (pct_err > max_err) max_err = pct_err;
  if (pct_err > 0.1) {
    std::cout << "C_FC_WEIGHT: " << std::endl;
    print_2d_tensor(decoder_mlp_c_fc_weight);
    std::cout << "C_FC_BIAS: " << std::endl;
    print_1d_tensor(decoder_mlp_c_fc_bias);
    std::cout << "C_PROJ_WEIGHT: " << std::endl;
    print_2d_tensor(decoder_mlp_c_proj_weight);
    std::cout << "C_PROJ_BIAS: " << std::endl;
    print_1d_tensor(decoder_mlp_c_proj_bias);
    std::cout << "Input block: " << std::endl;
    print_batched_2d_tensor(c_fc_out);
    std::cout << "Golden block: " << std::endl;
    print_batched_2d_tensor(decoder_after_mlp);
    std::cout << "Prose block: " << std::endl;
    print_batched_2d_tensor(prose_after_mlp);
    std::cout << "Error: " << diff.abs().max().item<float>() << std::endl;
    exit(1);
  }
  std::cout << "MLP passed" << std::endl;

  // add residual connection
  auto prose_after_residual_2 = torch::zeros({batch_size, seq_len, D}, torch::kFloat32);
  prose_float_wrapper::prose_matadd(prose_after_mlp.data_ptr<float>(),
                                    prose_after_ln_2.data_ptr<float>(),
                                    prose_after_residual_2.data_ptr<float>(),
                                    batch_size * seq_len * D);
  auto after_residual_connection_2 = decoder_after_mlp + decoder_after_ln_2;
  diff = after_residual_connection_2 - prose_after_residual_2;
  // assert error is less than 1e-5
  pct_err = diff.abs().max().item<float>() / after_residual_connection_2.abs().max().item<float>();
  if (pct_err > max_err) max_err = pct_err;
  if (pct_err > 0.1) {
    std::cout << "Input block: " << std::endl;
    print_batched_2d_tensor(after_residual_connection_2);
    std::cout << "Golden block: " << std::endl;
    print_batched_2d_tensor(prose_after_residual_2);
    std::cout << "Error: " << diff.abs().max().item<float>() << std::endl;
    exit(1);
  }
  std::cout << "Tests all passed" << std::endl;
#endif

  std::cout << "DONE!" << std::endl;
  std::cout << "Max Error: " << (max_err * 100) << "%" << std::endl;
  return 0;
}