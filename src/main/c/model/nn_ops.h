//
// Created by Christopher Kjellqvist on 4/22/24.
//

#ifndef C_EXEC_NN_OPS_H
#define C_EXEC_NN_OPS_H
//
// Created by Christopher Kjellqvist on 4/5/24.
//

#include <beethoven/fpga_handle.h>
#include <beethoven_hardware.h>
#include <beethoven/allocator/alloc.h>
#include "nn_ops.h"


namespace nn_ops {
  using namespace beethoven;
  intptr_t base_act_alloc = 0x3632000;
  static const remote_ptr NULL_RPTR
#ifdef BAREMETAL
(-1)
#endif
  ;

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
  static void prose_e_matmul(const remote_ptr &activations,
                             const remote_ptr &weights,
                             const remote_ptr &out,
                             const remote_ptr &bias,
                             const remote_ptr &norms,
                             int chosen_batch_size,
                             int M, int K, int N,
                             const remote_ptr &write_out);

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
  static void prose_m_matmul(const remote_ptr &activations,
                             const remote_ptr &weights,
                             const remote_ptr &out,
                             const remote_ptr &bias,
                             int chosen_batch_size,
                             int M,
                             int K,
                             int N,
                             bool output_transpose,
                             const remote_ptr &norms);

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
  static void prose_g_matmul(const remote_ptr &activations,
                             const remote_ptr &weights,
                             const remote_ptr &norm,
                             const remote_ptr &bias,
                             const remote_ptr &out,
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
  void prose_self_attention(
          const remote_ptr &input,
          const uint8_t &batch_size,
          const uint16_t &input_length,
          const uint16_t &D,
          const remote_ptr &query,
          const remote_ptr &key,
          const remote_ptr &value,
          const remote_ptr &query_bias,
          const remote_ptr &key_bias,
          const remote_ptr &value_bias,
          const remote_ptr temp[3],
          const remote_ptr &out);

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
  void prose_layer_norm(const remote_ptr &input,
                        const remote_ptr &gamma,
                        const remote_ptr &beta,
                        const uint8_t &batch_size,
                        const uint16_t &input_length,
                        const remote_ptr &out);
}
#endif //C_EXEC_NN_OPS_H
