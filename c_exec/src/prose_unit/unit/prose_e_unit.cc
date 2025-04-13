//
// Created by Christopher Kjellqvist on 1/8/24.
//

#include <beethoven/fpga_handle.h>
#include <beethoven_hardware.h>
#include <float_wrapper.h>
#include <variant>
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

//#include "prose_ops.h"

#ifndef BAREMETAL
std::random_device rd;
std::uniform_real_distribution<float> dist(-2, 2);
auto seed = 123; //rd();
std::default_random_engine eng(seed);
#endif
#ifndef BAREMETAL

float expwrap(float x) { return exp(x); }

#else

extern "C" {
// this uses 49kB less code than printf
void print_32bHex(uint32_t q) {
  for (int i = 7; i >= 0; --i) {
    char c = (q >> i * 4) & 0xF;
    if (c < 10) {
      UartPutc(c + 48);
    } else {
      UartPutc(c + 55);
    }
  }
}

void print_str(const char *s) {
  for (int i = 0; s[i] != 0; ++i) {
    UartPutc(s[i]);
  }
}
}
#endif

#include "prose_impl.h"

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
//      printf("answer %d(pre-activation)\n", i);
//      print_matrix(golden_matmul[i], M, Q);
    for (int j = 0; j < M * Q; ++j) {
      golden_pa_matmul[i][j] = golden_matmul[i][j];
      float lower = truncate_float_to_bf16_accuracy(golden_matmul[i][j]);
      float fapp = activation_f(lower);
      float fapplower = truncate_float_to_bf16_accuracy(fapp);
      golden_matmul[i][j] = fapplower;
    }

    // accumulate norms for golden pytorch_unit
    for (int j = 0; j < M; ++j) {
      golden_norms[i + j * chosen_batch_size] = 0;
      for (int k = 0; k < Q; ++k) {
        golden_norms[i + j * chosen_batch_size] += golden_matmul[i][j * Q + k];
      }
      golden_norms[i + j * chosen_batch_size] = float(1.0) / golden_norms[i + j * chosen_batch_size];
    }
  }

  auto write_out = handle.malloc(M * chosen_batch_size * sizeof(uint16_t));
  prose_impl::prose_e_matmul(prose_activation_transpose,
                             prose_weights,
                             prose_out,
                             nullptr,
                             nullptr,
                             PROSE_biasNONE,
                             chosen_batch_size, false, M, K, Q,
                             write_out, false);

  float *norm_vec = new float[M * chosen_batch_size];
  for (int i = 0; i < M * chosen_batch_size; ++i) {
    uint16_t p = ((uint16_t*)write_out.getHostAddr())[i];
    // std::cout << "read " << std::hex << "0x" << p << " from mem" << std::endl;
    auto q = ((uint32_t) p) << 16;
    norm_vec[i] = reinterpret_cast<float&>(q);
    // std::cout << "AH " << norm_vec[i] << std::endl;
  }

  uint16_t golden_cast[chosen_batch_size][M * Q];
  uint16_t golden_transpose[M * Q * chosen_batch_size];


  for (int b = 0; b < chosen_batch_size; ++b) {
    for (int i = 0; i < M * Q; ++i) {
      uint32_t r = reinterpret_cast<uint32_t &>(golden_matmul[b][i]);
      golden_cast[b][i] = r >> 16;
      if (r & 0x8000) golden_cast[b][i]++;
    }
  }

  convertRowMajorFormatToProSEColMajor((uint16_t *) golden_cast,
                                       golden_transpose, M, Q,
                                       chosen_batch_size, PROSE_Nmin);

  const std::function<bool(float, float)> &equality = generous_is_equal;
#ifndef BAREMETAL
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

#endif
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

//      handle.shutdown();
      exit(1);
    }
  }

  for (int i = 0; i < M * chosen_batch_size; ++i) {
    if (!super_permissive_equals(norm_vec[i], golden_norms[i], 0.05, 1)) {
      print_context();
      uint16_t norm_cast = reinterpret_cast<uint32_t &>(norm_vec[i]) >> 16;
      uint16_t gold_cast = reinterpret_cast<uint32_t &>(golden_norms[i]) >> 16;
      printf("Norm(%d) doesn't seem right. gold %0.4f(%04x) =/= obs %0.4f(%04x)\n",
             i, golden_norms[i], gold_cast, norm_vec[i], norm_cast);
      print_max_err();
      reset_error_measurement();
//      handle.shutdown();
      exit(1);
    }
  }
  for (int i = 0; i < chosen_batch_size; ++i) delete[] activation[i];
  for (int i = 0; i < chosen_batch_size; ++i) delete[] golden_matmul[i];
  for (int i = 0; i < chosen_batch_size; ++i) delete[] golden_pa_matmul[i];
  delete[] norm_vec;
}

#endif

int main() {
  int n_trials = 1;
#ifdef BAREMETAL
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

  int batch_size = 4;

  uint32_t deadbeef = peek_addr(BeethovenMMIOOffset + AXIL_DEBUG);
  print_32bHex(deadbeef);

  const int arsz = K * batch_size * 4 * PROSE_ECore_N;
  float norms[4 * PROSE_ECore_N * batch_size];
  prose_e_matmul(
          remote_ptr(0, nullptr),
          remote_ptr(arsz * 2, nullptr),
          remote_ptr(arsz * 4, nullptr),
          batch_size,
          2 * PROSE_ECore_N,
          K,
          2 * PROSE_ECore_N,
          (float *) norms);
  for (const auto &r: norms) {
    print_32bHex(reinterpret_cast<const uint32_t &>(r));
  }
  prose_e_matmul(
          remote_ptr(0, nullptr),
          remote_ptr(arsz * 2, nullptr),
          remote_ptr(arsz * 4, nullptr),
          batch_size,
          2 * PROSE_ECore_N,
          K,
          2 * PROSE_ECore_N,
          remote_ptr(arsz * 8, nullptr));


  UartEndSimulation();
#else
  printf("random seed: %x\n", seed);
  fflush(stdout);
  // test_prose_e(1, 64, PROSE_ECore_N * 1, PROSE_ECore_N * 1, expwrap);
  // std::cout << "PASSED" << std::endl;
  // test_prose_e(1, 64, PROSE_ECore_N * 1, PROSE_ECore_N * 2, expwrap);
  // std::cout << "PASSED" << std::endl;
  // test_prose_e(1, 64, PROSE_ECore_N * 2, PROSE_ECore_N * 1, expwrap);
  // std::cout << "PASSED" << std::endl;
  // test_prose_e(1, 64, PROSE_ECore_N * 2, PROSE_ECore_N * 2, expwrap);
  // std::cout << "PASSED" << std::endl;
  test_prose_e(1, 64, PROSE_ECore_N * NCORES_E * 2, PROSE_ECore_N * 4, expwrap);
  std::cout << "PASSED" << std::endl;
  printf("\n\nError statistics:\n");
  print_max_err();
  printf("shutting down\n");
  fflush(stdout);
//  handle.shutdown();
#endif
}

#else
int main() {
  printf("PROSE not enabled\n");
}
#endif
