//
// Created by Christopher Kjellqvist on 9/30/24.
//

#include "prose_impl.h"
#include "beethoven/rocc_cmd.h"
#include "beethoven_hardware.h"
#include <bit>

using namespace beethoven;

static uint8_t m_core_busy_bits = 0, e_core_busy_bits = 0, g_core_busy_bits = 0;

static const int M_FULL = (1 << NCORES_M) - 1;
static const int E_FULL = (1 << NCORES_E) - 1;
static const int G_FULL = (1 << NCORES_G) - 1;
static beethoven::response_handle<MCore::opErr> m_handles[NCORES_M];
static beethoven::response_handle<ECore::opErr> e_handles[NCORES_E];
static beethoven::response_handle<GCore::opErr> g_handles[NCORES_G];

static bool is_illegal(float f) {
  auto q = std::bit_cast<uint32_t>(f);
  return (q & 0x7F800000) == 0x7f800000;
}

static void check_illegal_ar(void *ar, size_t n) {
  for (auto i = 0; i < n; ++i) {
    if (is_illegal(((float*)ar)[i]))
      throw std::runtime_error("illegal at index " + std::to_string(i));
  }
}

#define NO_AVAIL 0xFFU
//#define NO_ERR_CHECK

static inline void alloc_bit(uint8_t &busy, uint8_t a) {
  busy |= 1 << a;
}

/**
 * The performance of this kernel is extremely important. Might be worth doing asm for.
 */
template<typename handle_t>
uint8_t poll(uint8_t &busy, beethoven::response_handle<handle_t> *ar, const int8_t &L) {
  uint8_t mask = 1;
  uint8_t avail = NO_AVAIL;
  std::optional<handle_t> q;
  for (uint8_t i = 0; i < L; ++i, mask <<= 1) {
    if (busy & mask) {
      if ((q = ar[i].try_get()).has_value()) {
        busy ^= mask;
        avail = i;
#ifndef NO_ERR_CHECK
        if (q.value().err != 0) {
          printf("Error: %d\n", q.value().err);
        }
#endif
      }
    } else {
      avail = i;
    }
  }
  return avail;
}

template<typename handle_t>
uint8_t barrier(uint8_t &busy, beethoven::response_handle<handle_t> *ar, const int8_t &L) {
  uint8_t mask = 1;
  uint8_t avail = NO_AVAIL;
  handle_t q;
  for (uint8_t i = 0; i < L; ++i, mask <<= 1) {
    if (busy & mask) {
      q = ar[i].get();
      busy ^= mask;
      avail = i;
#ifndef NO_ERR_CHECK
      if (q.err != 0) {
        printf("Error: %d\n", q.err);
      }
#endif
    } else {
      avail = i;
    }
  }
  return avail;
}


#define M_IS_FULL() m_core_busy_bits == M_FULL
#define G_IS_FULL() g_core_busy_bits == G_FULL
#define E_IS_FULL() e_core_busy_bits == E_FULL

#define VERBOSE(x)
//#define VERBOSE(x) x

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
void prose_impl::prose_e_matmul(remote_ptr const &activations,
                                remote_ptr const &weights,
                                remote_ptr const &out,
                                remote_ptr const *bias,
                                remote_ptr const *norms,
                                int biasMode,
                                int chosen_batch_size,
                                bool weights_are_batched,
                                int M, int K, int N,
                                remote_ptr const &write_out,
                                bool norm_per_batch) {
#ifdef PROSE_ECore_N
  const auto activation_stripe_size_bytes = K * PROSE_Nmin * 2 * chosen_batch_size;
  VERBOSE(assert(N % PROSE_Nmin == 0));
  VERBOSE(assert(M % PROSE_Nmin == 0));
  VERBOSE(assert(K <= PROSE_kMax));
  VERBOSE(assert(chosen_batch_size <= PROSE_maxBatch));
  auto output_stripe_sz_bytes = PROSE_ECore_N * N * 2 * chosen_batch_size;
  auto output_substripe_sz_bytes = PROSE_Nmin * N * 2 * chosen_batch_size;
  bool use_norms = norms != nullptr;
  /**  tiled matrix multiply **/
  // row tiles in output matrix
  int bias_addr_incr, bias_sz_bytes;
  if (biasMode == PROSE_biasNONE) {
    VERBOSE(std::cout << "disabled bias" << std::endl);
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
  int rows = M / PROSE_ECore_N;
  int cols_mo = N / PROSE_ECore_N - 1;
  auto act_acc = activations;
  auto out_acc = out;
  auto smax_acc = write_out;
  remote_ptr norm_acc;
  remote_ptr bias_acc;
  if (use_norms) norm_acc = *norms;
  if (biasMode != PROSE_biasNONE) bias_acc = *bias;
  for (int i = 0; i < rows; ++i) {
    uint8_t core_id;
    while ((core_id = poll(e_core_busy_bits, e_handles, NCORES_E)) == NO_AVAIL) {}
    e_handles[core_id] = ECore::matrixOp(core_id,
                                         act_acc,
                                         weights,
                                         out_acc,
                                         chosen_batch_size,
                                         bias_acc,
                                         biasMode,
                                         bias_sz_bytes,
                                         K,
                                         cols_mo,
                                         norm_acc,
                                         use_norms,
                                         norm_per_batch,
                                         output_substripe_sz_bytes,
                                         smax_acc,
                                         weights_are_batched);
    out_acc = out_acc + output_stripe_sz_bytes;
    act_acc = act_acc + activation_stripe_size_bytes * (PROSE_ECore_N / PROSE_Nmin);
    smax_acc = smax_acc + PROSE_ECore_N * 2 * chosen_batch_size;
    if (use_norms) norm_acc = norm_acc + PROSE_ECore_N * 2 * (norm_per_batch ? chosen_batch_size : 1);
    if (biasMode != PROSE_biasNONE) bias_acc = bias_acc + bias_addr_incr;
    alloc_bit(e_core_busy_bits, core_id);
  }
  barrier(e_core_busy_bits, e_handles, NCORES_E);

  check_illegal_ar(write_out.getHostAddr(), M);
  check_illegal_ar(out.getHostAddr(), M * N * chosen_batch_size);
#endif
}

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
void prose_impl::prose_m_matmul(remote_ptr const &activations,
                                remote_ptr const &weights,
                                remote_ptr const &out,
                                remote_ptr const *bias,
                                int biasMode,
                                int chosen_batch_size,
                                int M, int K, int N,
                                bool output_transpose,
                                remote_ptr const *norms,
                                bool weights_are_batched,
                                bool norm_per_batch) {
#ifdef PROSE_MCore_N
  const auto activation_stripe_size_bytes = K * PROSE_Nmin * 2 * chosen_batch_size;
  VERBOSE(assert(N % PROSE_Nmin == 0));
  VERBOSE(assert(M % PROSE_Nmin == 0));
  VERBOSE(assert(K <= PROSE_kMax));
  VERBOSE(assert(chosen_batch_size <= PROSE_maxBatch));
  auto output_substripe_sz_bytes = PROSE_Nmin * (output_transpose ? N : M) * 2 * chosen_batch_size;
  auto output_row_increment =
          2 * chosen_batch_size * PROSE_MCore_N * (output_transpose ? N : PROSE_Nmin);
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

  int rows = M / PROSE_MCore_N;
  int cols_mo = N / PROSE_MCore_N - 1;
  auto act_acc = activations;
  auto out_acc = out;
  remote_ptr norm_acc;
  remote_ptr bias_acc;
  if (use_norms) norm_acc = *norms;
  if (biasMode != PROSE_biasNONE) bias_acc = *bias;

  for (int i = 0; i < rows; ++i) {
    //printf("STATUS:"
    //       "\tbusy: %x\n",
    //       m_core_busy_bits);
    uint8_t core_id;
    while ((core_id = poll(m_core_busy_bits, m_handles, NCORES_M)) == NO_AVAIL) {}
    m_handles[core_id] = MCore::matrixOp(core_id,
                                         act_acc,
                                         weights,
                                         out_acc,
                                         chosen_batch_size,
                                         bias_acc,
                                         biasMode,
                                         bias_sz_bytes,
                                         K,
                                         cols_mo,
                                         norm_acc,
                                         use_norms,
                                         norm_per_batch,
                                         output_substripe_sz_bytes,
                                         output_transpose,
                                         weights_are_batched);
    out_acc = out_acc + output_row_increment;
    act_acc = act_acc + activation_stripe_size_bytes * (PROSE_MCore_N / PROSE_Nmin);
    if (use_norms) norm_acc = norm_acc + PROSE_ECore_N * 2 * (norm_per_batch ? chosen_batch_size : 1);
    if (biasMode != PROSE_biasNONE) bias_acc = bias_acc + bias_addr_incr;
    alloc_bit(m_core_busy_bits, core_id);
//    printf("setting m busy idx %d\n", core_id);
  }
  barrier(m_core_busy_bits, m_handles, NCORES_M);
  check_illegal_ar(out.getHostAddr(), M * N * chosen_batch_size);
#endif
}

void prose_impl::prose_g_matmul(remote_ptr const &activations,
                                remote_ptr const &weights,
                                remote_ptr const *norms,
                                remote_ptr const *bias,
                                remote_ptr const &out,
                                int chosen_batch_size,
                                int M, int K, int N,
                                bool norm_per_batch,
                                int biasMode) {
#ifdef PROSE_GCore_N
  const auto activation_stripe_size_bytes = K * PROSE_Nmin * 2 * chosen_batch_size;
  const auto weight_stripe_size_bytes = K * PROSE_Nmin * 2;
  VERBOSE(assert((N % PROSE_Nmin) == 0));
  VERBOSE(assert((M % PROSE_Nmin) == 0));
  VERBOSE(assert(K <= PROSE_kMax));
  VERBOSE(assert(chosen_batch_size <= PROSE_maxBatch));
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

  int rows = M / PROSE_GCore_N;
  int cols_mo = N / PROSE_GCore_N - 1;

  auto act_acc = activations;
  auto out_acc = out;
  remote_ptr norm_acc;
  remote_ptr bias_acc;
  if (use_norms) norm_acc = *norms;
  if (biasMode != PROSE_biasNONE) bias_acc = *bias;

  for (int i = 0; i < rows; ++i) {
    uint8_t core_id;
    while ((core_id = poll(g_core_busy_bits, g_handles, NCORES_G)) == NO_AVAIL) {}
    g_handles[core_id] = GCore::matrixOp(core_id,
                                         act_acc,
                                         weights,
                                         out_acc,
                                         chosen_batch_size,
                                         bias_acc,
                                         biasMode,
                                         bias_sz_bytes,
                                         K,
                                         cols_mo,
                                         norm_acc,
                                         use_norms,
                                         norm_per_batch,
                                         output_substripe_sz_bytes,
                                         true,
                                         false);
    out_acc = out_acc + output_stripe_sz_bytes;
    act_acc = act_acc + activation_stripe_size_bytes * (PROSE_GCore_N / PROSE_Nmin);
    if (use_norms) norm_acc = norm_acc + PROSE_ECore_N * 2 * (norm_per_batch ? chosen_batch_size : 1);
    if (biasMode != PROSE_biasNONE) bias_acc = bias_acc + bias_addr_incr;
    alloc_bit(g_core_busy_bits, core_id);
  }
  barrier(g_core_busy_bits, g_handles, NCORES_G);
#endif
}

void prose_impl::prose_layer_norm(
        const beethoven::remote_ptr &input,
        const beethoven::remote_ptr &gamma_beta,
        const uint8_t &batch_size,
        const uint16_t &input_length,
        const uint16_t &seq_len,
        const beethoven::remote_ptr &out) {
  float norm = 1.F / float(input_length);
  uint32_t norm_fp = reinterpret_cast<uint32_t &>(norm);
  Norm::norm(0, gamma_beta, input, batch_size * PROSE_Nmin, (seq_len / PROSE_Nmin), norm_fp >> 16, flagLayerNorm, out,
             true,
             input_length).get();
  auto pptr = (uint16_t*)out.getHostAddr();
  std::cout << "lnorm" << std::endl;
  for (int i = 0; i < 32; ++i) {
	  std::cout << i << " 0x" << std::hex << pptr[i] << std::endl;
		  if (pptr[i] == 0x7fc0) throw std::runtime_error("got illegal");
  }
  std::cout << "\n\n";
}

void prose_impl::prose_matadd(const beethoven::remote_ptr &a,
                              const beethoven::remote_ptr &b,
                              const beethoven::remote_ptr &c,
                              const uint32_t &length) {
  MatrixAdd::MatAdd(0, a, b, c, length).get();

  std::cout << "add" << std::endl;
  auto pptr = (uint16_t*)c.getHostAddr();
  for (int i = 0; i < 32; ++i) {
	  std::cout << i << " 0x" << std::hex << pptr[i] << std::endl;
		  if (pptr[i] == 0x7fc0) throw std::runtime_error("got illegal");
  }
  std::cout << "\n\n";
}
