//
// Created by Christopher Kjellqvist on 9/30/24.
//

#include "auto_allocate.h"
#include "beethoven/rocc_cmd.h"
#include "beethoven_hardware.h"
#include "prose_rptr.h"
#include "prose_rptr_structured.h"

using namespace beethoven;

/**
 * @brief Perform a matrix multiplication operation on the input tensors with
 * exponentiation activation We perform the operation: out = exp(norm *
 * (activations * weights) + bias)
 * @param activations input tensor
 * @param weights weights tensor
 * @param out output tensor
 * @param bias per-row pre-activation bias tensor
 * @param norms per-row pre-activation normalization tensor
 * @param chosen_batch_size chosen batch size
 * @param M # of rows in the input tensor (activation)
 * @param K # of columns in the input tensor (activation) and # of rows in the
 * weights tensor
 * @param N # of columns in the weights tensor
 * @param write_out output tensor for the normalization values provided by
 * softmax
 */
void prose_e_matmul(remote_ptr const &activations, remote_ptr const &weights,
                    remote_ptr const &out, remote_ptr const *bias,
                    remote_ptr const *norms, int biasMode,
                    int chosen_batch_size, bool weights_are_batched, int M,
                    int K, int N, remote_ptr const &write_out,
                    bool norm_per_batch) {
  const auto activation_stripe_size_bytes =
      K * PROSE_Nmin * 2 * chosen_batch_size;
  auto output_stripe_sz_bytes = PROSE_ECore_N * N * 2 * chosen_batch_size;
  auto output_substripe_sz_bytes = PROSE_Nmin * N * 2 * chosen_batch_size;
  bool use_norms = norms != nullptr;
  /**  tiled matrix multiply **/
  // row tiles in output matrix
  int bias_addr_incr, bias_sz_bytes;
  if (biasMode == PROSE_biasNONE) {
  } else if (biasMode == PROSE_biasCOLS) {
    bias_sz_bytes = N * 2;
    bias_addr_incr = 0;
  } else if (biasMode == PROSE_biasMATRIX) {
    bias_sz_bytes = PROSE_Nmin * N * 2;
    bias_addr_incr = bias_sz_bytes * (PROSE_ECore_N / PROSE_Nmin);
  } else if (biasMode == PROSE_biasBATCHEDMATRIX) {
    bias_sz_bytes = PROSE_Nmin * N * chosen_batch_size * 2;
    bias_addr_incr = bias_sz_bytes * (PROSE_ECore_N / PROSE_Nmin);
  }
  int row_execs_to_do = M / PROSE_ECore_N;
  int cols_mo = N / PROSE_ECore_N - 1;
  auto act_acc = activations;
  auto out_acc = out;
  auto smax_acc = write_out;
  remote_ptr norm_acc;
  remote_ptr bias_acc;
  if (use_norms)
    norm_acc = *norms;
  if (biasMode != PROSE_biasNONE)
    bias_acc = *bias;
  ECore::matrixOp(0, act_acc, weights, out_acc, chosen_batch_size, bias_acc,
                  biasMode, bias_sz_bytes, K, cols_mo, row_execs_to_do - 1,
                  norm_acc, use_norms, norm_per_batch,
                  output_substripe_sz_bytes, smax_acc, weights_are_batched)
      .get();
}

/**
 * @brief Perform a matrix multiplication operation on the input tensors with NO
 * activation We perform the operation: out = norm * (activations * weights) +
 * bias
 * @param activations
 * @param weights
 * @param out
 * @param bias
 * @param chosen_batch_size
 * @param M
 * @param K
 * @param N
 * @param output_transpose
 * @param norms
 */
void prose_m_matmul(const remote_ptr &activations, const remote_ptr &weights,
                    const remote_ptr &out, remote_ptr const *bias, int biasMode,
                    int chosen_batch_size, int M, int K, int N,
                    bool output_transpose, const remote_ptr *norms,
                    bool weights_are_batched, bool norm_per_batch) {
  const auto activation_stripe_size_bytes =
      K * PROSE_Nmin * 2 * chosen_batch_size;
  auto output_substripe_sz_bytes =
      PROSE_Nmin * (output_transpose ? N : M) * 2 * chosen_batch_size;
  auto output_row_increment = 2 * chosen_batch_size * PROSE_MCore_N *
                              (output_transpose ? N : PROSE_Nmin);
  /**  tiled matrix multiply **/
  // row tiles in output matrix
  bool use_norms = norms != nullptr;
  int bias_addr_incr, bias_sz_bytes;
  if (biasMode == PROSE_biasNONE) {
  } else if (biasMode == PROSE_biasCOLS) {
    bias_sz_bytes = N * 2;
    bias_addr_incr = 0;
  } else if (biasMode == PROSE_biasMATRIX) {
    bias_sz_bytes = PROSE_Nmin * N * 2;
    bias_addr_incr = bias_sz_bytes * (PROSE_MCore_N / PROSE_Nmin);
  } else if (biasMode == PROSE_biasBATCHEDMATRIX) {
    bias_sz_bytes = PROSE_Nmin * N * chosen_batch_size * 2;
    bias_addr_incr = bias_sz_bytes * (PROSE_MCore_N / PROSE_Nmin);
  }

  int rows_to_do = M / PROSE_MCore_N;
  int cols_mo = N / PROSE_MCore_N - 1;
  auto act_acc = activations;
  auto out_acc = out;
  remote_ptr norm_acc;
  remote_ptr bias_acc;
  if (use_norms)
    norm_acc = *norms;
  if (biasMode != PROSE_biasNONE)
    bias_acc = *bias;

  MCore::matrixOp(0, act_acc, weights, out_acc, chosen_batch_size, bias_acc,
                  biasMode, bias_sz_bytes, K, cols_mo, rows_to_do - 1, norm_acc,
                  use_norms, norm_per_batch, output_substripe_sz_bytes,
                  output_transpose, weights_are_batched)
      .get();
}

void prose_g_matmul(remote_ptr const &activations, remote_ptr const &weights,
                    remote_ptr const *norms, remote_ptr const *bias,
                    remote_ptr const &out, int chosen_batch_size, int M, int K,
                    int N, bool norm_per_batch, int biasMode) {
  const auto activation_stripe_size_bytes =
      K * PROSE_Nmin * 2 * chosen_batch_size;
  const auto weight_stripe_size_bytes = K * PROSE_Nmin * 2;
  bool use_norms = norms != nullptr;

  auto output_stripe_sz_bytes = PROSE_GCore_N * N * 2 * chosen_batch_size;
  auto output_substripe_sz_bytes = PROSE_Nmin * N * 2 * chosen_batch_size;
  /**  tiled matrix multiply **/
  // row tiles in output matrix
  int bias_addr_incr, bias_sz_bytes;
  if (biasMode == PROSE_biasNONE) {
  } else if (biasMode == PROSE_biasCOLS) {
    bias_sz_bytes = N * 2;
    bias_addr_incr = 0;
  } else if (biasMode == PROSE_biasMATRIX) {
    bias_sz_bytes = PROSE_Nmin * N * 2;
    bias_addr_incr = bias_sz_bytes * (PROSE_GCore_N / PROSE_Nmin);
  } else if (biasMode == PROSE_biasBATCHEDMATRIX) {
    bias_sz_bytes = PROSE_Nmin * N * chosen_batch_size * 2;
    bias_addr_incr = bias_sz_bytes * (PROSE_GCore_N / PROSE_Nmin);
  }

  int row_execs_to_do = M / PROSE_GCore_N;
  int cols_mo = N / PROSE_GCore_N - 1;

  auto act_acc = activations;
  auto out_acc = out;
  remote_ptr norm_acc;
  remote_ptr bias_acc;
  if (use_norms)
    norm_acc = *norms;
  if (biasMode != PROSE_biasNONE)
    bias_acc = *bias;

  GCore::matrixOp(0, act_acc, weights, out_acc, chosen_batch_size, bias_acc,
                  biasMode, bias_sz_bytes, K, cols_mo, row_execs_to_do - 1,
                  norm_acc, use_norms, norm_per_batch,
                  output_substripe_sz_bytes, true, false)
      .get();
}

void prose_layer_norm(const beethoven::remote_ptr &input,
                      const beethoven::remote_ptr &gamma_beta,
                      const uint8_t &batch_size, const uint16_t &input_length,
                      const uint16_t &seq_len,
                      const beethoven::remote_ptr &out) {
  float norm = 1.F / float(input_length);
  uint32_t norm_fp = reinterpret_cast<uint32_t &>(norm);
  Norm::norm(0, gamma_beta, input, batch_size * PROSE_Nmin,
             (seq_len / PROSE_Nmin), norm_fp >> 16, flagLayerNorm, out, true,
             input_length)
      .get();
}

void prose_matadd(const beethoven::remote_ptr &a,
                  const beethoven::remote_ptr &b,
                  const beethoven::remote_ptr &c, const uint32_t &length) {
  MatrixAdd::MatAdd(0, a, b, c, length).get();
}

constinit TransformerLayer layers[2] = {TransformerLayer(0),
                                        TransformerLayer(1)};

void prose_self_attention(const remote_ptr &input, uint8_t batch_size,
                          uint16_t input_length, uint16_t D, uint16_t head_size,
                          const remote_ptr out_accumulation, int t_id, int layer_id) {
  const remote_ptr(&temps)[4] =
      my_prose_allocations.selfatten_intermediates[t_id];
  const TransformerLayer layer = layers[layer_id];
  // QUERY PROJECTION
  prose_m_matmul(input, layer.qproj_w, temps[0], nullptr, PROSE_biasNONE,
                 batch_size, input_length, D, head_size, true, nullptr, false,
                 false);
  // KEY PROJECTION
  prose_m_matmul(input, layer.kproj_w, temps[1], nullptr, PROSE_biasNONE,
                 batch_size, input_length, D, head_size,
                 false /* DOUBLE CHECK: Use non-transpose output here*/,
                 nullptr, false, false);
  auto &attention_score_matrix_temp =
      my_prose_allocations.selfatten_attenscore[t_id];
  // SOFTMAX(QUERY X KEY^T)
  prose_e_matmul(temps[0], temps[1], attention_score_matrix_temp, &MASK,
                 nullptr /* DOUBLE CHECK: GPTNeo doesn't use scaled attention*/,
                 PROSE_biasMATRIX, batch_size, true, input_length, head_size,
                 input_length, temps[3], false);

  // VALUE PROJECTION
  prose_m_matmul(input, layer.vproj_w, temps[2], nullptr, PROSE_biasNONE,
                 batch_size, input_length, D, head_size,
                 true, // transpose (yes) - same reason as previous NOTE
                 nullptr,
                 false, // weights are batched (no)
                 false  // norm-per-batch (no)
  );

  // ATTENTION OUTPUT
  prose_m_matmul(attention_score_matrix_temp, temps[2], temps[1], nullptr,
                 PROSE_biasNONE, batch_size, input_length, input_length,
                 head_size, true, &temps[3],
                 true, // normalize per batch with softmax norm factors
                 true  // weights are batched
  );

  //  auto temp_  = new float[batch_size * input_length * head_size];
  //  convertPCMtoTCM((uint16_t *) temps[1].getHostAddr(), temp_, input_length,
  //  head_size, batch_size, PROSE_Nmin); auto temp_tensor =
  //  torch::from_blob(temp_, {batch_size, input_length, head_size}); std::cout
  //  << "temp_tensor sizes: " << temp_tensor.sizes() << std::endl;
  //  print_batched_2d_tensor(temp_tensor);
  //  auto out_accumulation_host = new float[batch_size * input_length * D];
  //  std::cout << "PROSE BEFORE UPDATE: " << std::endl;
  //  handle.copy_from_fpga(out_accumulation);
  //  convertPCMtoTCM((uint16_t *) out_accumulation.getHostAddr(),
  //  out_accumulation_host, input_length, D, batch_size, PROSE_Nmin); auto
  //  out_accumulation_tensor = torch::from_blob(out_accumulation_host,
  //  {batch_size, input_length, D});
  //  print_batched_2d_tensor(out_accumulation_tensor);

  // on the first go-through we initialize the output accumulator with the
  // matmul + bias, then we accumulate on it
  prose_m_matmul(temps[1], layer.oproj_w, out_accumulation, &layer.oproj_b,
                 PROSE_biasCOLS, batch_size, input_length, head_size, D, true,
                 nullptr, false, false);
  //
  //  std::cout << "PROSE AFTER UPDATE: " << std::endl;
  //  handle.copy_from_fpga(out_accumulation);
  //  convertPCMtoTCM((uint16_t *) out_accumulation.getHostAddr(),
  //  out_accumulation_host, input_length, D, batch_size, PROSE_Nmin);
  //  out_accumulation_tensor = torch::from_blob(out_accumulation_host,
  //  {batch_size, input_length, D});
  //  print_batched_2d_tensor(out_accumulation_tensor);
}

// void prose_full_model(int num_layers, int batch_size, int seq_len, int
// embed_dim,
//   beethoven::remote_ptr &if_input) {
//   for (int layer_num = 0; layer_num < num_layers; ++layer_num) {
//     std::vector<int64_t> head_dims = {embed_dim / num_heads, embed_dim};
//     for (int h = 0; h < num_heads; ++h) {
//       auto if_qproj =
//           get_text_head(attn.attention.q_proj.weight, layer_num, h,
//           head_dims);
//       auto if_kproj =
//           get_text_head(attn.attention.k_proj.weight, layer_num, h,
//           head_dims);
//       auto if_vproj =
//           get_text_head(attn.attention.v_proj.weight, layer_num, h,
//           head_dims);
//       auto if_oproj = get_text_head(attn.attention.out_proj.weight,
//       layer_num,
//                                     h, head_dims);
//       qarray[h] = if_qproj.data;
//       karray[h] = if_kproj.data;
//       varray[h] = if_vproj.data;
//       oarray[h] = if_oproj.data;
//     }
//     std::vector<int64_t> embed_dims = {embed_dim, embed_dim};
//     std::vector<int64_t> embed_vec = {embed_dim};
//     std::vector<int64_t> causal_dim = {1, seq_len, seq_len};
//     std::vector<int64_t> input_dim = {1, seq_len, embed_dim};
//     auto if_oprojb =
//         get_text(attn.attention.out_proj.bias, layer_num, embed_vec);
//     auto if_causal =
//         get_text(attn.attention.causal_mask, layer_num, {1, seq_len,
//         seq_len});
//     auto if_gout = get_text(output, layer_num, input_dim);
//     auto if_ln1_w = get_text(ln_1.weight, layer_num, embed_vec);
//     auto if_ln1_b = get_text(ln_1.bias, layer_num, embed_vec);
//     auto if_ln2_w = get_text(ln_2.weight, layer_num, embed_vec);
//     auto if_ln2_b = get_text(ln_2.bias, layer_num, embed_vec);
//     auto if_mlp_fc_wgt =
//         get_text(mlp.c_fc.weight, layer_num, {embed_dim, 3072});
//     auto if_mlp_fc_bias = get_text(mlp.c_fc.bias, layer_num, {3072});
//     auto if_mlp_proj_wgt =
//         get_text(mlp.c_proj.weight, layer_num, {3072, embed_dim});
//     auto if_mlp_proj_bias = get_text(mlp.c_proj.bias, layer_num,
//     {embed_dim});
//     float *output = new float[batch_size * seq_len * embed_dim];
//     memset(output, 0, sizeof(float) * batch_size * seq_len * embed_dim);
//     std::cout << "INPUT TO DECODER: " << std::endl;
//     for (int i = 0; i < seq_len; ++i) {
//       for (int j = 0; j < embed_dim; ++j) {
//         float q = hw_rolling_acc[i * embed_dim + j];
//         printf("%0.4f ", q);
//       }
//       printf("\n");
//     }
//     decoder_layer(
//         hw_rolling_acc, if_ln1_w.data, if_ln1_b.data, if_ln2_w.data,
//         if_ln2_b.data, qarray, karray, varray, oarray, if_oprojb.data,
//         if_causal.data, if_mlp_fc_wgt.data, if_mlp_fc_bias.data,
//         if_mlp_proj_wgt.data, if_mlp_proj_bias.data, batch_size, seq_len,
//         embed_dim, num_heads, output);
//     memcpy(hw_rolling_acc, output,
//            batch_size * embed_dim * seq_len * sizeof(float));
//   }
//   auto hw_final_ln_wgt = interchange_format::from_float_file(
//       get_text_checkpoint_dir() + "/transformer.ln_f.weight.float",
//       {embed_dim});
//   auto hw_final_ln_bias = interchange_format::from_float_file(
//       get_text_checkpoint_dir() + "/transformer.ln_f.bias.float",
//       {embed_dim});
//   prose_float_wrapper::prose_layer_norm(hw_rolling_acc, hw_final_ln_wgt.data,
//                                         hw_final_ln_bias.data, batch_size,
//                                         seq_len, embed_dim, hw_rolling_acc);
//   auto if_output = interchange_format::from_float_file(
//       get_text_checkpoint_dir() + "/output.float",
//       {batch_size, seq_len, embed_dim});
//   auto q = interchange_format(hw_rolling_acc, {batch_size, seq_len,
//   embed_dim},
//                               batch_size * seq_len * embed_dim);
// #endif
// }
