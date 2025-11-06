#ifndef PROSE_LIB_H
#define PROSE_LIB_H

#ifdef LOCAL
#include <beethoven/allocator/alloc.h>
#else
#include <beethoven_baremetal/allocator/alloc_baremetal.h>
#endif
#include <beethoven_hardware.h>
#include <coroutine>

//
// Created by Christopher Kjellqvist on 9/30/24.
//

#include "beethoven/rocc_cmd.h"
using namespace beethoven;

#ifdef LOCAL
// expect that there's the input file in ../../model/gpt_neo/prose_input.bin
remote_ptr get_from_float_file(uint64_t offset, uint64_t len);
#define PTR_FROM_OFFSET(off, len) (get_from_float_file(off, len))
#define __ptr_annot__ const
#define __constructor_annot__
#else
#define PTR_FROM_OFFSET(off, len) (beethoven::remote_ptr(off))
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

struct promise;

struct prose_thread : std::coroutine_handle<promise>
{
    using promise_type = ::promise;

    bool done() const noexcept;
    void resume() noexcept;
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
prose_thread prose_e_matmul_nb(remote_ptr const &activations, remote_ptr const &weights,
                    remote_ptr const &out, remote_ptr const *bias,
                    remote_ptr const *norms, int biasMode,
                    int chosen_batch_size, bool weights_are_batched, int M,
                    int K, int N, remote_ptr const &write_out,
                    bool norm_per_batch, const dec_dep* dep, const int head_id);
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
prose_thread prose_m_matmul_nb(remote_ptr const &activations, remote_ptr const &weights,
                    remote_ptr const &out, remote_ptr const *bias, int biasMode,
                    int chosen_batch_size, int M, int K, int N,
                    bool output_transpose, remote_ptr const *norms,
                    bool weights_are_batched, bool norm_per_batch, M_type which_m, dec_dep* dep, const int head_id,
                    int stripe_stride = 1);

prose_thread prose_g_matmul_nb(remote_ptr const &activations, remote_ptr const &weights,
                    remote_ptr const *norms, remote_ptr const *bias,
                    remote_ptr const &out, int chosen_batch_size, int M, int K,
                    int N, bool norm_per_batch, int biasMode, dec_dep* dep);

prose_thread prose_layer_norm_nb(const beethoven::remote_ptr &input,
                      const beethoven::remote_ptr &gamma_beta,
                      const uint8_t &batch_size, const uint16_t &input_length,
                      const uint16_t &seq_len,
                      const beethoven::remote_ptr &out, dec_dep* dep, DecoderMask which_norm);

prose_thread prose_matadd_nb(const beethoven::remote_ptr &a,
                  const beethoven::remote_ptr &b,
                  const beethoven::remote_ptr &c, const uint32_t &length,
                dec_dep* dep, DecoderMask which_add);


prose_thread prose_decoder_nb(const remote_ptr &input, const ModelConfig &config,
                   const remote_ptr &out_accumulation, int t_id, int layer_id, dec_dep* dep, DecoderMask which_norm);
#endif
