#ifndef PROSE_LIB_H
#define PROSE_LIB_H

#include <beethoven_baremetal/allocator/alloc_baremetal.h>
#include <beethoven_hardware.h>

//
// Created by Christopher Kjellqvist on 9/30/24.
//

#include "beethoven/rocc_cmd.h"
#include "beethoven_hardware.h"

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
                    bool weights_are_batched, bool norm_per_batch);

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

void prose_self_attention(const remote_ptr &input, uint8_t batch_size,
                          uint16_t input_length, uint16_t D, uint16_t head_size,
                          const remote_ptr out_accumulation, int t_id,
                          int layer_id);
#endif