//
// Created by Christopher Kjellqvist on 1/8/24.
//

#include <beethoven/fpga_handle.h>
#include <beethoven_hardware.h>

#ifndef BAREMETAL

#include <random>
#include "util.h"

#else
#include "CMSDK_CM0.h"
#include "core_cm0.h"
#include "uart_stdout.h"
#endif

#ifdef PROSE
using namespace beethoven;
fpga_handle_t handle;

#include <prose_impl.h>

#ifndef BAREMETAL
std::random_device rd;
std::uniform_real_distribution<float> dist(-2, 2);
auto seed = rd();
std::default_random_engine eng(seed);
#endif

// A x B
// A matrix is M by K
// B matrix is K by Q
// K is the **shared** dimension


/**
 * row x col
 * activations are R x K, weights are K x C
 */

#ifndef BAREMETAL

void test_prose_e(int chosen_batch_size,
                  int K,
                  int M,
                  int Q,
                  std::function<float(float)> activation_f) {
  assert(chosen_batch_size <= PROSE_maxBatch);

  float *activation[chosen_batch_size];
  float weights[Q * K];
  float *golden_matmul[chosen_batch_size];
  float *golden_pa_matmul[chosen_batch_size];
  for (int i = 0; i < chosen_batch_size; ++i) activation[i] = new float[M * K];
  for (int i = 0; i < chosen_batch_size; ++i) golden_matmul[i] = new float[Q * M];
  for (int i = 0; i < chosen_batch_size; ++i) golden_pa_matmul[i] = new float[Q * M];

  float golden_norms[M * chosen_batch_size];

  auto prose_activation_transpose = handle.malloc(M * K * 2 * chosen_batch_size);
  auto prose_weights = handle.malloc(Q * K * 2 * chosen_batch_size);
  auto prose_out = handle.malloc(M * Q * 2 * chosen_batch_size);

  uint16_t bfloat_act[chosen_batch_size][M * K];
  uint16_t bfloat_wgt[Q * K];

  auto prose_host_act = (uint16_t *) prose_activation_transpose.getHostAddr();
  auto prose_host_wgt = (uint16_t *) prose_weights.getHostAddr();

  // initialize activations
  for (int b = 0; b < chosen_batch_size; ++b) {
    for (int i = 0; i < K * M; ++i) {
      auto act_raw = dist(eng);
      uint32_t act_hex = reinterpret_cast<uint32_t &>(act_raw);
      act_hex &= 0xffff0000;
      float act = reinterpret_cast<float &>(act_hex);
      activation[b][i] = act;
      bfloat_act[b][i] = uint16_t(act_hex >> 16);
    }
  }
  convertRowMajorFormatToProSEColMajor((uint16_t *) bfloat_act, prose_host_act,
                                       M, K, chosen_batch_size, PROSE_Nmin);


  // initialize weights
  for (int i = 0; i < K * Q; ++i) {
    auto wgt_raw = dist(eng);
    uint32_t wgt_hex = reinterpret_cast<uint32_t &>(wgt_raw);
    wgt_hex &= 0xffff0000;
    float wgt = reinterpret_cast<float &>(wgt_hex);
    weights[i] = wgt;
    bfloat_wgt[i] = uint16_t(wgt_hex >> 16);
  }
  convertRowMajorFormatToProSERowMajor(bfloat_wgt, prose_host_wgt, K, Q, 1, PROSE_Nmin);

  for (int i = 0; i < chosen_batch_size; ++i) {
    matmul(activation[i], weights, golden_matmul[i], M, Q, K, 1);
    for (int j = 0; j < M * Q; ++j) {
      golden_pa_matmul[i][j] = golden_matmul[i][j];
      float lower = truncate_float_to_bf16_accuracy(golden_matmul[i][j]);
      float fapp = activation_f(lower);
      float fapplower = truncate_float_to_bf16_accuracy(fapp);
      golden_matmul[i][j] = fapplower;
    }

    for (int j = 0; j < M; ++j) {
      golden_norms[i + j * chosen_batch_size] = 0;
      for (int k = 0; k < Q; ++k) {
        golden_norms[i + j * chosen_batch_size] += golden_matmul[i][j * Q + k];
      }
      golden_norms[i + j * chosen_batch_size] = float(1.0) / golden_norms[i + j * chosen_batch_size];
    }
  }

  prose_impl::prose_g_matmul(prose_activation_transpose,
                             prose_weights,
                             nullptr,
                             nullptr,
                             prose_out,
                             chosen_batch_size,
                             M, K, Q, false, PROSE_biasNONE);

  uint16_t golden_cast[chosen_batch_size][M * Q];
  uint16_t golden_transpose[M * Q * chosen_batch_size];


  for (int b = 0; b < chosen_batch_size; ++b) {
    for (int i = 0; i < M * Q; ++i) {
      uint32_t r = reinterpret_cast<uint32_t &>(golden_matmul[b][i]);
      golden_cast[b][i] = r >> 16;
      if (r & 0x8000) golden_cast[b][i]++;
    }
  }
  convertRowMajorFormatToProSEColMajor((uint16_t *) golden_cast, golden_transpose, M, Q,
                                       chosen_batch_size, PROSE_Nmin);

  for (int i = 0; i < chosen_batch_size; ++i) {
  }
  auto print_context = [&]() {
    printf("prose_weights\n");
    print_matrix(bfloat_wgt, K, Q);

    for (int b = 0; b < chosen_batch_size; ++b) {
      printf("activations %d\n", b);
      print_matrix(activation[b], M, K);
    }
    printf("weights\n");
    print_matrix(weights, K, Q);
    for (int i = 0; i < chosen_batch_size; ++i) {
      printf("answer (pre-act) %d\n", i);
      print_matrix(golden_pa_matmul[i], M, Q);
    }
    for (int i = 0; i < chosen_batch_size; ++i) {
      printf("answer %d\n", i);
      print_matrix(golden_matmul[i], M, Q);
    }
  };

  auto out_ar = (uint16_t *) prose_out.getHostAddr();
  for (int i = 0; i < M * Q * chosen_batch_size; ++i) {
    uint32_t gt = golden_transpose[i];
    uint32_t oa = out_ar[i];
    gt <<= 16;
    oa <<= 16;
    float gt_cast = reinterpret_cast<float &>(gt);
    float oa_cast = reinterpret_cast<float &>(oa);
    if (!super_permissive_equals(gt_cast, oa_cast, 0.05, 0)) {
      print_context();
      printf("%d doesn't seem right. gold %0.4f(%04x) =/= obs %0.4f(%04x)\n",
             i,
             gt_cast, golden_transpose[i],
             oa_cast, out_ar[i]);

      handle.shutdown();
      exit(1);
    }
  }
  for (int i = 0; i < chosen_batch_size; ++i) {
    delete[] activation[i];
    delete[] golden_matmul[i];
  }
}

#endif

#ifdef BAREMETAL
// this uses 49kB less code than printf
void print_32bHex(uint32_t q) {
  for (int i = 0; i < 8; ++i) {
    // 48 is '0' in ASCII
    UartPutc((q & 0xF) + 48);
    q >>= 4;
  }
} 

// big endian print out hex
#endif

int main() {
  int n_trials = 100;
#ifdef BAREMETAL
  int K = 16;
  // UART init
  UartStdOutInit();

  NVIC_DisableIRQ(TIMER0_IRQn);
  NVIC_DisableIRQ(TIMER1_IRQn);
  CMSDK_TIMER1->RELOAD = 0x1000;
  CMSDK_TIMER1->VALUE = 0xffffffff;
  CMSDK_TIMER1->CTRL = 0x01; /* Set enable */
// this takes 5kB... Let's use fputc if we really need this
//  printf("Hello world\n");
  const char hw[] = "Hello World!\n";
  for (int i = 0; i < sizeof(hw); ++i) {
    UartPutc(hw[i]);
  }

  // End simulation
  const int arsz = K * PROSE_maxBatch * 4 * PROSE_ECore_N;
  float norms[4 * PROSE_ECore_N * PROSE_maxBatch];
  prose_e_matmul(
          remote_ptr(0, arsz * 2),
          remote_ptr(arsz * 2, arsz * 2),
          remote_ptr(arsz * 4, arsz * 2),
          PROSE_maxBatch,
          4 * PROSE_ECore_N,
          K,
          4 * PROSE_ECore_N,
          norms);
  for (const auto &r: norms) {
    print_32bHex(reinterpret_cast<const uint32_t&>(r));
// printf("0x%08x\n", reinterpret_cast<const uint32_t&>(r));
  }

  UartEndSimulation();
#else
  printf("random seed: %x\n", seed);
  fflush(stdout);
  bool only_once = true;
  for (int K = 16; K <= PROSE_kMax; K += 16) {
    for (int j = 1; j <= 4; ++j) {
      for (int k = 1; k <= 4; ++k) {
        for (int batch = 1; batch <= PROSE_maxBatch; ++batch) {
          printf("\rExecuting: M_mult(%d), Q_mult(%d), batch(%d)", j, k, 1);
          fflush(stdout);
          for (int trial = 0; trial < n_trials; ++trial) {
            test_prose_e(batch, K, PROSE_GCore_N * j, PROSE_GCore_N * k, gelu);
            if (only_once) {
              goto break_out;
            }
          }
        }
      }
    }
  }
  break_out:
  printf("\n\nError statistics:\n");
  print_max_err();
  printf("shutting down\n");
  fflush(stdout);
#endif
}

#else
int main() {
  printf("PROSE not enabled\n");
}
#endif
