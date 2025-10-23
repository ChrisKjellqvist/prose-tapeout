//
// Created by Christopher Kjellqvist on 9/30/24.
//

#include "auto_allocate.h"
#ifdef LOCAL
#include "beethoven/fpga_handle.h"
#else
#include "beethoven_baremetal/fpga_handle.h"
#endif
#include "beethoven/rocc_cmd.h"
#include "beethoven_hardware.h"
#include "prose_rptr.h"
#include "prose_vec_rptr.h"
#include <cstring>

using namespace beethoven;
fpga_handle_t handle;

#ifndef LOCAL
const constinit AllLayers all_layers;
const constinit prose_allocations<1, 768, 1, 16, 12> my_prose_allocations =
    auto_alloc::get_prose_allocs<1, 768, 1, 16, 12>();
#else
#include <sys/mman.h>
#include <unistd.h>
remote_ptr get_from_float_file(uint64_t offset, uint64_t len) {
  FILE *f = fopen("../../model/gpt_neo/prose_input.raw", "r");
  if (f == nullptr) {
    throw std::runtime_error("Cannot open ../../model/gpt_neo/prose_input.raw");
  }
  auto fptr =
      mmap(nullptr, len, PROT_READ, MAP_SHARED | MAP_FILE, fileno(f), offset);
  remote_ptr ptr = handle.malloc(len);
  memcpy(ptr.getHostAddr(), fptr, len);
  munmap(fptr, len);
  return ptr;
}
AllLayers all_layers;
prose_allocations<1, 768, 1, 16, 12> my_prose_allocations;
void init_alloc() {
  my_prose_allocations = auto_alloc::get_prose_allocs<1, 768, 1, 16, 12>();
}
#endif
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
                    bool weights_are_batched, bool norm_per_batch,
                    int stripe_stride) {
  const auto activation_stripe_size_bytes =
      K * PROSE_Nmin * 2 * chosen_batch_size;
  auto output_substripe_sz_bytes =
      (PROSE_Nmin * (output_transpose ? N : M) * 2 * chosen_batch_size) *
      stripe_stride;
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

void prose_mh_self_attention(const remote_ptr &input, const remote_ptr &out,
                             const ModelConfig &config, int t_id,
                             int layer_id) {
#ifdef LOCAL
    printf("i\n");
    for (int i = 0; i < 10; ++i) {
      printf("%04x ", ((uint16_t *)input.getHostAddr())[i]);
    }
    printf("\n");
#endif

  const remote_ptr(&temps)[4] =
      my_prose_allocations.selfatten_intermediates[t_id];
  auto &attention_score_matrix_temp =
      my_prose_allocations.selfatten_attenscore[t_id];
  const TransformerLayer &layer = all_layers.layers[layer_id];
  for (int head_idx = 0; head_idx < config.n_heads; ++head_idx) {
    // QUERY PROJECTION
    prose_m_matmul(input, layer.proj_wgts[head_idx].qproj, temps[0], nullptr,
                   PROSE_biasNONE, config.batch_size, config.seq_len, config.D,
                   config.head_size, true, nullptr, false, false);
#ifdef LOCAL
    printf("qproj h%d\n", head_idx);
    for (int i = 0; i < 10; ++i) {
      printf("%04x ", ((uint16_t *)temps[0].getHostAddr())[i]);
    }
    printf("\n");
#endif

    // KEY PROJECTION
    prose_m_matmul(input, layer.proj_wgts[head_idx].kproj, temps[1], nullptr,
                   PROSE_biasNONE, config.batch_size, config.seq_len, config.D,
                   config.head_size,
                   false /* DOUBLE CHECK: Use non-transpose output here*/,
                   nullptr, false, false);
#ifdef LOCAL
    printf("kproj h%d\n", head_idx);
    for (int i = 0; i < 10; ++i) {
      printf("%04x ", ((uint16_t *)temps[1].getHostAddr())[i]);
    }
    printf("\n");
#endif

    // SOFTMAX(QUERY X KEY^T)
    prose_e_matmul(
        temps[0], temps[1], attention_score_matrix_temp,
        &all_layers.layers[layer_id].causal_mask,
        nullptr /* DOUBLE CHECK: GPTNeo doesn't use scaled attention*/,
        PROSE_biasMATRIX, config.batch_size, true, config.seq_len,
        config.head_size, config.seq_len, temps[3], false);
#ifdef LOCAL
    printf("e h%d\n", head_idx);
    for (int i = 0; i < 10; ++i) {
      printf("%04x ",
             ((uint16_t *)attention_score_matrix_temp.getHostAddr())[i]);
    }
    printf("\n");
#endif

    // VALUE PROJECTION
    prose_m_matmul(input, layer.proj_wgts[head_idx].vproj, temps[2], nullptr,
                   PROSE_biasNONE, config.batch_size, config.seq_len, config.D,
                   config.head_size,
                   true, // transpose (yes) - same reason as previous NOTE
                   nullptr,
                   false, // weights are batched (no)
                   false  // norm-per-batch (no)
    );
#ifdef LOCAL
    printf("vproj h%d\n", head_idx);
    for (int i = 0; i < 10; ++i) {
      printf("%04x ", ((uint16_t *)temps[2].getHostAddr())[i]);
    }
    printf("\n");
#endif

    // ATTENTION OUTPUT
    prose_m_matmul(attention_score_matrix_temp, temps[2], temps[1], nullptr,
                   PROSE_biasNONE, config.batch_size, config.seq_len,
                   config.seq_len, config.head_size, true, &temps[3],
                   true, // normalize per batch with softmax norm factors
                   true, // weights are batched
                   12    // stripe stride will hop across all of the other heads

    );
#ifdef LOCAL
    printf("score h%d\n", head_idx);
    for (int i = 0; i < 10; ++i) {
      printf("%04x ", ((uint16_t *)temps[1].getHostAddr())[i]);
    }
    printf("\n");
#endif
  }
  prose_m_matmul(temps[1], layer.oproj_w, out, &layer.oproj_b, PROSE_biasCOLS,
                 config.batch_size, config.seq_len, config.head_size, config.D,
                 true, nullptr, false, false);
#ifdef LOCAL
  printf("oproj\n");
  for (int i = 0; i < 10; ++i) {
    printf("%04x ", ((uint16_t *)out.getHostAddr())[i]);
  }
  printf("\n");
#endif
}

void prose_decoder(const remote_ptr &input, const remote_ptr &out,
                   const ModelConfig &config, int t_id, int layer_id) {
  const auto &residual = input;
#ifdef VERBOSE
  {
    uint32_t f_i = *(uint16_t *)input.getHostAddr();
    f_i <<= 16;
    float f = std::bit_cast<float>(f_i);
    printf("input: %0.4f\n", f);
  }
#endif
  Norm::norm(0, all_layers.layers[layer_id].ln1_wb, input, 1, config.batch_size,
             1.0 / 768, flagLayerNorm, my_prose_allocations.ln_out[t_id], 1,
             config.D)
      .get();
#ifdef VERBOSE
  {
    uint32_t f_i = *(uint16_t *)input.getHostAddr();
    f_i <<= 16;
    float f = std::bit_cast<float>(f_i);
    printf("ln1: %0.4f\n", f);
  }
#endif

  prose_mh_self_attention(my_prose_allocations.ln_out[t_id], out, config, t_id,
                          layer_id);
  MatrixAdd::MatAdd(0, residual, out, residual,
                    config.D * config.batch_size * config.seq_len)
      .get();
#ifdef VERBOSE
  {
    uint32_t f_i = *(uint16_t *)input.getHostAddr();
    f_i <<= 16;
    float f = std::bit_cast<float>(f_i);
    printf("atn: %0.4f\n", f);
  }
#endif

  Norm::norm(0, all_layers.layers[layer_id].ln2_wb, residual, 1,
             config.batch_size, 1.0 / 768, flagLayerNorm,
             my_prose_allocations.ln_out[t_id], 1, config.D)
      .get();
#ifdef VERBOSE
  {
    uint32_t f_i = *(uint16_t *)input.getHostAddr();
    f_i <<= 16;
    float f = std::bit_cast<float>(f_i);
    printf("ln2: %0.4f\n", f);
  }
#endif

  // START MLP
  prose_g_matmul(my_prose_allocations.ln_out[t_id],
                 all_layers.layers[layer_id].mlp_fc_w, nullptr,
                 &all_layers.layers[layer_id].mlp_fc_b,
                 my_prose_allocations.mlp_intermediate[t_id], config.batch_size,
                 config.seq_len, config.D, config.D * 4, 0, PROSE_biasCOLS);

  prose_m_matmul(my_prose_allocations.mlp_intermediate[t_id],
                 all_layers.layers[layer_id].mlp_proj_w,
                 my_prose_allocations.ln_out[t_id],
                 &all_layers.layers[layer_id].mlp_proj_w, PROSE_biasCOLS,
                 config.batch_size, config.seq_len, config.D * 4, config.D,
                 true, nullptr, false, false);
// END MLP
#ifdef VERBOSE
  {
    uint32_t f_i = *(uint16_t *)input.getHostAddr();
    f_i <<= 16;
    float f = std::bit_cast<float>(f_i);
    printf("mlp: %0.4f\n", f);
  }
#endif

  MatrixAdd::MatAdd(0, residual, my_prose_allocations.ln_out[t_id], out,
                    config.D * config.batch_size * config.seq_len)
      .get();
#ifdef VERBOSE
  {
    uint32_t f_i = *(uint16_t *)input.getHostAddr();
    f_i <<= 16;
    float f = std::bit_cast<float>(f_i);
    printf("out: %0.4f\n", f);
  }
#endif
}
