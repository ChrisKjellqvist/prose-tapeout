//
// Created by Entropy Xu on 9/9/24.
//

#ifndef PROSE_COMPILER_PROSE_ISA_GOLDEN_H
#define PROSE_COMPILER_PROSE_ISA_GOLDEN_H

#include "stdint.h"
//#include "torch/extension.h"


#ifdef USE_TORCH
namespace prose_isa_golden {
  /**
   * @brief Perform a matrix multiplication operation on the input tensors with exponentiation activation
   * We perform the operation: out = exp(norm * (activations * weights) + bias)
   * @param activations input tensor
   * @param weights weights tensor
   * @param out output tensor
   * @param bias per-row pre-activation bias tensor
   * @param norms per-row pre-activation normalization tensor
   * @param chosen_batch_size chosen batch size
   * @param M # of rows in the input tensor (activation)
   * @param K # of columns in the input tensor (activation) and # of rows in the weights tensor
   * @param N # of columns in the weights tensor
   * @param write_out output tensor for the normalization values provided by softmax
   */
  void prose_e_matmul(float* activations,
                             float* weights,
                             float* out,
                             float* bias,
                             float* norms,
                             int chosen_batch_size,
                             int M, int K, int N,
                             float* write_out);

  /**
   * @brief Perform a matrix multiplication operation on the input tensors with NO activation
   * We perform the operation: out = norm * (activations * weights) + bias
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
  void prose_m_matmul(float* activations,
                             float* weights,
                             float* out,
                             float* bias,
                             int chosen_batch_size,
                             int M,
                             int K,
                             int N,
                             bool output_transpose,
                             float* norms);

  /**
   * @brief Perform a matrix multiplication operation on the input tensors with GeLU activation
   * We perform the operation: out = GeLU(norm * (activations * weights) + bias)
   * @param activations input tensor
   * @param weights weights tensor
   * @param norm per-row pre-activation normalization tensor
   * @param bias per-row pre-activation bias tensor
   * @param out output tensor
   * @param chosen_batch_size chosen batch size
   * @param output_transpose whether to transpose the output. Importantly, if the norm and bias are set to valid pointers,
   *   then the output should be transposed. Otherwise, the bias and normalization will be applied by-column instead of
   *   by-row. Ask me for clarification if this doesn't make sense.
   * @param M # of rows in the input tensor (activation)
   * @param K # of columns in the input tensor (activation) and # of rows in the weights tensor
   * @param N # of columns in the weights tensor
   */
  void prose_g_matmul(float* activations,
                             float* weights,
                             float* norm,
                             float* bias,
                             float* out,
                             int chosen_batch_size,
                             int M, int K, int N);

  /**
   * @brief Perform a self attention operation on the input tensor. This does not use kv-caching, that needs to be
   * done manually using the other operations.
   *
   * The input shape is (batch_size, input_length, D). The output shape is (batch_size, input_length, input_length).
   *
   * @param input The input tensor.
   * @param batch_size The batch size of the input tensor.
   * @param input_length The number of features in the input tensor.
   * @param D The dimension of the tensor
   * @param query The query weights
   * @param key The key weights
   * @param value The value weights
   * @param query_bias The query bias
   * @param key_bias The key bias
   * @param value_bias The value bias
   * @param temp The temporary tensors: We need space to compute in * Kw, in * Vw, K * V, etc...
   * If we need more space for temporary tensor, we can modify this, of course.
   * @param out The output tensor.
   */
  void prose_self_attention(float*  input,
                            uint8_t batch_size,
                            uint16_t input_length,
                            uint16_t D,
                            uint16_t head_size,
                            float*  query,
                            float*  key,
                            float*  value,
                            float*  out_proj,
                            float*  query_bias,
                            float*  key_bias,
                            float*  value_bias,
                            float*  causal_mask,
                            float*  out_accumulation);

  /**
   * @brief Perform a layer normalization operation on the input tensor.
   * Arguments correspond to https://pytorch.org/docs/stable/generated/torch.nn.LayerNorm.html
   * @param input The input tensor.
   * @param gamma
   * @param beta
   * @param batch_size
   * @param input_length
   * @param out normalized output tensor
   */
  void prose_layer_norm(float *input, float *gamma, float *beta,
                        uint8_t batch_size, uint16_t input_length, uint16_t D,
                        float *out);

  void prose_multi_head_attention(
          float* input,
          float* q_proj_w,
          float* k_proj_w,
          float* v_proj_w,
          float* o_proj_w,
          float* o_proj_b,
          float* causal_mask,
          uint8_t batch_size,
          uint16_t seq_len,
          uint16_t embed_dim,
          uint16_t num_heads,
          float* attn_output
  );

  void prose_mlp(float *input,
         float *fc_wgt,
         float *fc_bias,
         float *proj_wgt,
         float *proj_bias,
         float *out,
         int batch_size, int seq_len, int embed_size);

  void prose_matadd(float* a, float* b, float* c, int sz);

  void decoder_layer(
          float *input,
          float *ln1_wgt, float *ln1_bias,
          float *ln2_wgt, float *ln2_bias,
          float *qarray, float *karray, float *varray,
          float *oarray, float *obias,
          float *causal_mask,
          float *mlp_fc_wgt, float *mlp_fc_bias,
          float *mlp_proj_wgt, float *mlp_proj_bias,
          uint8_t batch_size, uint16_t seq_len, uint16_t embed_dim, uint16_t num_heads,
          float *attn_output);
};
#endif
#endif //PROSE_COMPILER_PROSE_ISA_GOLDEN_H