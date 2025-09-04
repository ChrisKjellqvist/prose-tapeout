//
// Created by Christopher Kjellqvist on 9/30/24.
//

#ifndef PROSE_COMPILER_prose_float_wrapper_H
#define PROSE_COMPILER_prose_float_wrapper_H

#include "beethoven/fpga_handle.h"
#include <variant>

extern beethoven::fpga_handle_t handle;

namespace prose_float_wrapper {

  void prose_multi_head_attention(
          float *input,
          float **q_proj_w,
          float **k_proj_w,
          float **v_proj_w,
          float **o_proj_w,
          float *o_proj_b,
          float *causal_mask,
          uint8_t batch_size,
          uint16_t seq_len,
          uint16_t embed_dim,
          uint16_t num_heads,
          float *attn_output);

  void prose_self_attention(float *input,
                            uint8_t batch_size, uint16_t input_length,
                            uint16_t D, uint16_t head_size,
                            float *query, float *key,
                            float *value, float *query_bias,
                            float *key_bias, float *value_bias,
                            float *attention_mask,

                            float *sqrt_vector,
                            float *output_bias,
                            float *output_proj_tensor,

                            bool use_output_bias,
                            beethoven::remote_ptr out_accumulation);

  void prose_e_matmul(
          std::variant<uint16_t *, float *, beethoven::remote_ptr> activations,
          std::variant<uint16_t *, float *, beethoven::remote_ptr> weights,
          std::variant<uint16_t *, float *, beethoven::remote_ptr> out,
          std::variant<float *, beethoven::remote_ptr> bias,// instead of a bias vector, this is now a matrix
          float *norms,
          bool weights_are_batched,
          int chosen_batch_size, int M, int K, int N,
          std::variant<float *, beethoven::remote_ptr> write_out,
          int biasMode,
          bool norm_per_batch);

  void prose_m_matmul(std::variant<uint16_t *, float *, beethoven::remote_ptr> activations,
                      std::variant<uint16_t *, float *, beethoven::remote_ptr> weights,
                      std::variant<uint16_t *, float *, beethoven::remote_ptr> out,
                      std::variant<float *, beethoven::remote_ptr> bias,
                      std::variant<float *, beethoven::remote_ptr> norms,
                      int chosen_batch_size,
                      int M, int K, int N,
                      bool output_transpose,
                      bool norm_per_batch,
                      bool weights_are_batched,
                      int biasMode);

  void prose_g_matmul(std::variant<uint16_t *, float *, beethoven::remote_ptr> activations,
                      std::variant<uint16_t *, float *, beethoven::remote_ptr> weights,
                      std::variant<uint16_t *, float *, beethoven::remote_ptr> out,
                      std::variant<float *, beethoven::remote_ptr> bias,
                      float *norms,
                      int biasMode,
                      int chosen_batch_size,
                      int M, int K, int N,
                      bool per_batch_norm);

  void prose_layer_norm(float *input, float *gamma, float *beta,
                        uint8_t batch_size,
                        uint16_t seq_len,
                        uint16_t input_length,
                        float *out);

  void prose_matadd(float *input1, float *input2, float *output, uint32_t eles);
}// namespace prose_float_wrapper

#endif//PROSE_COMPILER_prose_float_wrapper_H