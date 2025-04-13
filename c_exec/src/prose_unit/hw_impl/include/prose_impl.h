//
// Created by Christopher Kjellqvist on 9/30/24.
//

#ifndef PROSE_COMPILER_PROSE_IMPL_H
#define PROSE_COMPILER_PROSE_IMPL_H
#include "beethoven/fpga_handle.h"

namespace prose_impl {
  using namespace beethoven;

  void prose_e_matmul(remote_ptr const &activations,
                      remote_ptr const &weights,
                      remote_ptr const &out,
                      remote_ptr const *bias,
                      remote_ptr const *norms,
                      int biasMode,
                      int chosen_batch_size,
                      bool weights_are_batched,
                      int M, int K, int N,
                      remote_ptr const &write_out,
                      bool norm_per_batch);

  void prose_m_matmul(remote_ptr const &activations,
                      remote_ptr const &weights,
                      remote_ptr const &out,
                      remote_ptr const *bias,
                      int biasMode,
                      int chosen_batch_size,
                      int M, int K, int N,
                      bool output_transpose,
                      remote_ptr const *norms,
                      bool weights_are_batched,
                      bool norm_per_batch);

  void prose_g_matmul(remote_ptr const &activations,
                      remote_ptr const &weights,
                      remote_ptr const *norms,
                      remote_ptr const *bias,
                      remote_ptr const &out,
                      int chosen_batch_size,
                      int M, int K, int N,
                      bool per_batch_norm,
                      int biasMode);

  void prose_layer_norm(const beethoven::remote_ptr &input,
                        const beethoven::remote_ptr &gamma_beta,
                        const uint8_t &batch_size,
                        const uint16_t &input_length,
                        const uint16_t &seq_len,
                        const beethoven::remote_ptr &out);

  void prose_matadd(const beethoven::remote_ptr &a,
                    const beethoven::remote_ptr &b,
                    const beethoven::remote_ptr &c,
                    const uint32_t &length);
};// namespace prose_impl

#endif//PROSE_COMPILER_PROSE_IMPL_H