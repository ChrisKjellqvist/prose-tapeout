// //
// // Created by Christopher Kjellqvist on 4/5/24.
// //

// #include <beethoven/fpga_handle.h>
// #include <beethoven_hardware.h>
// #include <beethoven/allocator/alloc.h>
// #include "nn_ops.h"
// #include "prose_rptr.h"

// using namespace beethoven;

// namespace nn_ops {
// //  static const remote_ptr NULL_RPTR(-1);

//   static void prose_e_matmul(const remote_ptr &activations,
//                              const remote_ptr &weights,
//                              const remote_ptr &out,
//                              int chosen_batch_size,
//                              int M, int K, int Q,
//                              const remote_ptr &write_out) {
//     const auto activation_stripe_size_bytes = K * PROSE_Nmin * 2 * chosen_batch_size;
//     const auto weight_stripe_size_bytes = K * PROSE_Nmin * 2;
//     auto output_stripe_sz_bytes = PROSE_Nmin * Q * 2 * chosen_batch_size;
//     for (int i = 0; i < M / PROSE_ECore_N; ++i) {
//       ECore::activationPrime(0, activations + activation_stripe_size_bytes * i, chosen_batch_size, K).get();
//       for (int j = 0; j < Q / PROSE_ECore_N; ++j) {
//         auto wgt_p = weights + weight_stripe_size_bytes * j;
//         remote_ptr out_addr;
//         out_addr = out +
//                    i * output_stripe_sz_bytes +
//                    j * PROSE_Nmin * PROSE_Nmin * 2 * chosen_batch_size;
//         ECore::matrixOp(0, // core_id
//                         wgt_p, // pointer to B array
//                         out_addr, // pointer to output tile
//                         false, // not matrix add
//                         output_stripe_sz_bytes,
//                         j == 0, // start of new row
//                         false // don't use row normalization
//         ).get();
//       }
//       ECore::softmax_norm_fetch2(0, 1, write_out).get();
//     }
//   }

//   static void prose_m_matmul(const remote_ptr &activations,
//                              const remote_ptr &weights,
//                              const remote_ptr &out,
//                              const remote_ptr &bias,
//                              int chosen_batch_size,
//                              int M,
//                              int K,
//                              int Q,
//                              bool output_transpose,
//                              const remote_ptr &norms) {
//     const auto activation_stripe_size_bytes = K * PROSE_Nmin * 2 * chosen_batch_size;
//     const auto weight_stripe_size_bytes = K * PROSE_Nmin * 2;
//     auto output_stripe_sz_bytes = PROSE_Nmin * (output_transpose ? Q : M) * 2 * chosen_batch_size;
//     bool use_norms = !(norms == NULL_RPTR);
//     for (int i = 0; i < M / PROSE_MCore_N; ++i) {
//       MCore::activationPrime(0, activations + activation_stripe_size_bytes * i, chosen_batch_size, K).get();
//       if (use_norms) {
//         MCore::normInit(0, norms + i * chosen_batch_size * 2 * PROSE_MCore_N, chosen_batch_size).get();
//       }
//       for (int j = 0; j < Q / PROSE_MCore_N; ++j) {
//         auto wgt_p = weights + weight_stripe_size_bytes * j;
//         remote_ptr out_addr;
//         if (output_transpose)
//           out_addr = out +
//                      i * output_stripe_sz_bytes +
//                      j * PROSE_Nmin * PROSE_Nmin * 2 * chosen_batch_size;
//         else
//           out_addr = out +
//                      j * output_stripe_sz_bytes +
//                      i * PROSE_Nmin * PROSE_Nmin * 2 * chosen_batch_size;
//         MCore::matrixOp(0, // core_id
//                         wgt_p, // pointer to B array
//                         out_addr, // pointer to output tile
//                         false, // not matrix add
//                         output_stripe_sz_bytes,
//                         output_transpose,
//                         j == 0, // start of new row
//                         use_norms // don't use row normalization
//         ).get();
//       }
//     }
//   }

//   static void prose_g_matmul(const remote_ptr &activations, const remote_ptr &weights, const remote_ptr &out,
//                              int chosen_batch_size,
//                              bool output_transpose,
//                              int M, int K, int Q) {
//     const auto activation_stripe_size_bytes = K * PROSE_Nmin * 2 * chosen_batch_size;
//     const auto weight_stripe_size_bytes = K * PROSE_Nmin * 2;
//     auto output_stripe_sz_bytes = PROSE_Nmin * (output_transpose ? Q : M) * 2 * chosen_batch_size;
//     for (int i = 0; i < M / PROSE_GCore_N; ++i) {
//       GCore::activationPrime(0, activations + activation_stripe_size_bytes * i, chosen_batch_size, K).get();
//       for (int j = 0; j < Q / PROSE_GCore_N; ++j) {
//         auto wgt_p = weights + weight_stripe_size_bytes * j;
//         remote_ptr out_addr;
//         if (output_transpose)
//           out_addr = out +
//                      i * output_stripe_sz_bytes +
//                      j * PROSE_Nmin * PROSE_Nmin * 2 * chosen_batch_size;
//         else
//           out_addr = out +
//                      j * output_stripe_sz_bytes +
//                      i * PROSE_Nmin * PROSE_Nmin * 2 * chosen_batch_size;
//         GCore::matrixOp(0, // core_id
//                         wgt_p, // pointer to B array
//                         out_addr, // pointer to output tile
//                         false, // not matrix add
//                         output_stripe_sz_bytes,
//                         output_transpose,
//                         j == 0, // start of new row
//                         false // don't use row normalization
//         ).get();
//       }
//     }
//   }

//   void prose_self_attention(
//           const remote_ptr &input,
//           const uint8_t &batch_size,
//           const uint16_t &input_length,
//           const remote_ptr &query,
//           const remote_ptr &key,
//           const remote_ptr &value,
//           const remote_ptr &query_bias,
//           const remote_ptr &key_bias,
//           const remote_ptr &value_bias,
//           const remote_ptr temp[3],
//           const remote_ptr &out);
// }