#ifndef PROSE_LIB_H
#define PROSE_LIB_H

#ifdef LOCAL
#include <beethoven/allocator/alloc.h>
#include <beethoven/fpga_handle.h>
#else
#include <beethoven_baremetal/allocator/alloc_baremetal.h>
#include <beethoven_baremetal/fpga_handle.h>
#endif
#include <beethoven_hardware.h>

//
// Created by Christopher Kjellqvist on 9/30/24.
//

#include "beethoven/rocc_cmd.h"
using namespace beethoven;

extern fpga_handle_t handle;

#ifdef LOCAL
// expect that there's the input file in ../../model/gpt_neo/prose_input.bin
remote_ptr get_from_float_file(uint64_t offset, uint64_t len);
#define PTR_FROM_OFFSET_C(off, len) = (get_from_float_file(off, len))
#define PTR_FROM_OFFSET_H(off, len) ;
#define __ptr_annot__ extern
#define __constructor_annot__
#else
#define PTR_FROM_OFFSET_H(off, len) = (beethoven::remote_ptr(off))
#define __constructor_annot__ constexpr
#define __ptr_annot__ constexpr
#endif

struct ModelConfig {
  const int batch_size = -1, D = -1, n_heads = -1, head_size = -1,
            n_layers = -1, norm_type = -1, seq_len = -1;
  static constexpr ModelConfig GPTNeoConfig(int batch_size, int seq_len) {
    return ModelConfig(batch_size, 768, 12, 64, 12, flagLayerNorm, seq_len);
  }
  constexpr ~ModelConfig() = default;
  constexpr ModelConfig() = default;

private:
  constexpr ModelConfig(int batch_size, int D, int n_heads, int head_size,
                        int n_layers, int norm_type, int seq_len)
      : batch_size(batch_size), D(D), n_heads(n_heads), head_size(head_size),
        n_layers(n_layers), norm_type(norm_type), seq_len(seq_len) {}
};

struct ProjLayer {
  beethoven::remote_ptr kproj;
  beethoven::remote_ptr vproj;
  beethoven::remote_ptr qproj;
  __constructor_annot__ ProjLayer() {}
  __constructor_annot__ ~ProjLayer() {}
};

struct TransformerLayer {
  static const int n_layers = 12;

  ProjLayer proj_wgts[n_layers];
  beethoven::remote_ptr ln1_wb;
  beethoven::remote_ptr oproj_w, oproj_b;
  beethoven::remote_ptr ln2_wb;
  beethoven::remote_ptr mlp_fc_w, mlp_fc_b;
  beethoven::remote_ptr mlp_proj_w, mlp_proj_b;
  beethoven::remote_ptr causal_mask;
  __constructor_annot__ TransformerLayer() {}
  __constructor_annot__ ~TransformerLayer() {}
};

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
                    bool norm_per_batch);
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
void prose_m_matmul(remote_ptr const &activations, remote_ptr const &weights,
                    remote_ptr const &out, remote_ptr const *bias, int biasMode,
                    int chosen_batch_size, int M, int K, int N,
                    bool output_transpose, remote_ptr const *norms,
                    bool weights_are_batched, bool norm_per_batch,
                    int stripe_stride = 1);

void prose_g_matmul(remote_ptr const &activations, remote_ptr const &weights,
                    remote_ptr const *norms, remote_ptr const *bias,
                    remote_ptr const &out, int chosen_batch_size, int M, int K,
                    int N, bool norm_per_batch, int biasMode);

void prose_layer_norm(const beethoven::remote_ptr &input,
                      const beethoven::remote_ptr &gamma_beta,
                      const uint8_t &batch_size, const uint16_t &input_length,
                      const uint16_t &seq_len,
                      const beethoven::remote_ptr &out);

void prose_matadd(const beethoven::remote_ptr &a,
                  const beethoven::remote_ptr &b,
                  const beethoven::remote_ptr &c, const uint32_t &length);

void prose_mh_self_attention(const remote_ptr &input, const remote_ptr &out,
                             const ModelConfig &config, int t_id, int layer_id);

void prose_decoder(const remote_ptr &input, const remote_ptr &out,
                   const ModelConfig &config, int t_id, int layer_id);
#endif
