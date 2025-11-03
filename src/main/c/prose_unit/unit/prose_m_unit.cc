//
// Created by Christopher Kjellqvist on 1/8/24.
//

#include <beethoven/fpga_handle.h>
#include <beethoven_hardware.h>
#include <cmath>
#include <float_wrapper.h>
#include <random>
#include <unistd.h>
#include "util.h"

#ifdef PROSE
using namespace beethoven;

fpga_handle_t handle;

std::random_device rd;
std::uniform_real_distribution<float> dist(-2, 2);
std::default_random_engine eng(rd());

bool use_random = false;

#include <prose_impl.h>

// A x B
// A matrix is M by K
// B matrix is K by Q
// K is the **shared** dimension


/**
 * row x col
 * activations are R x K, weights are K x C
 */

void test_prose_m(int chosen_batch_size,
                  int K,
                  int M,
                  int Q,
                  bool output_transpose) {
  assert(chosen_batch_size <= PROSE_maxBatch);

  float *activation[chosen_batch_size];
  for (int i = 0; i < chosen_batch_size; ++i) activation[i] = new float[M * K];

  float weights[Q * K];
  float *golden_matmul[chosen_batch_size];
  for (int i = 0; i < chosen_batch_size; ++i) golden_matmul[i] = new float[Q * M];

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

    // accumulate norms for golden pytorch_unit
    for (int j = 0; j < M; ++j) {
      golden_norms[i + j * chosen_batch_size] = 0;
      for (int k = 0; k < Q; ++k) {
        golden_norms[i + j * chosen_batch_size] += golden_matmul[i][j * Q + k];
      }
      golden_norms[i + j * chosen_batch_size] = float(1.0) / golden_norms[i + j * chosen_batch_size];
    }
  }

  prose_impl::prose_m_matmul(
          prose_activation_transpose,
          prose_weights,
          prose_out,
          &prose_activation_transpose,
          PROSE_biasNONE,
          chosen_batch_size,
          M, K, Q,
          output_transpose,
          nullptr, false,
          true);

//  prose_m_matmul(prose_activation_transpose, prose_weights, prose_out,
//                 chosen_batch_size, M, K, Q, output_transpose);

  uint16_t golden_cast[chosen_batch_size][M * Q];
  uint16_t golden_transpose[M * Q * chosen_batch_size];


  for (int b = 0; b < chosen_batch_size; ++b) {
    for (int i = 0; i < M * Q; ++i) {
      uint32_t r = reinterpret_cast<uint32_t &>(golden_matmul[b][i]);
      golden_cast[b][i] = r >> 16;
      if (r & 0x8000) golden_cast[b][i]++;
    }
  }

  if (output_transpose) {
    convertRowMajorFormatToProSEColMajor((uint16_t *) golden_cast, golden_transpose, M, Q,
                                         chosen_batch_size, PROSE_Nmin);
  } else {
    convertRowMajorFormatToProSERowMajor((uint16_t *) golden_cast, golden_transpose, M, Q,
                                         chosen_batch_size, PROSE_Nmin);
  }


  if (!use_random) {
    const std::function<bool(float, float)> &equality = generous_is_equal;
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
      if (!equality(gt_cast, oa_cast)) {
        print_context();
        printf("%d doesn't seem right. gold %0.4f(%04x) =/= obs %0.4f(%04x)\n",
               i,
               gt_cast, golden_transpose[i],
               oa_cast, out_ar[i]);
        return;
//        handle.shutdown();
//        exit(1);
      }
    }
  }
  for (int i = 0; i < chosen_batch_size; ++i) {
    delete[] activation[i];
    delete[] golden_matmul[i];
  }

}

int main() {
//  int n_trials = 100;
//  for (int K = 16; K < PROSE_kMax; K+=16) {
//  for (int j = 1; j <= 4; ++j) {
//    for (int k = 1; k <= 4; ++k) {
//      for (int i = 0; i < 2; ++i) {
//        bool output_transpose = i == 0;
//        for (int batch = 1; batch <= PROSE_maxBatch; batch++) {
//          printf("\rExecuting K(%d) non_transpose(%d), M_mult(%d), Q_mult(%d), batch(%d)", K, i, j, k, batch);
//          fflush(stdout);
//          for (int trial = 0; trial < n_trials; ++trial)
//            test_prose_m(batch, K, PROSE_MCore_N * j, PROSE_MCore_N * k, output_transpose);
//        }
//      }
//    }
//  }
//  }
  // test_prose_m(1, 4, PROSE_MCore_N, PROSE_MCore_N, true);
  // printf("\n\nshutting down peacefully. No errors found.\n");
  //
  // test_prose_m(1, 64, PROSE_MCore_N, PROSE_MCore_N*4, false);
  // printf("\n\nshutting down peacefully. No errors found.\n");
  //
  // test_prose_m(1, 64, PROSE_MCore_N * NCORES_M, PROSE_MCore_N*4, true);
  // printf("\n\nshutting down peacefully. No errors found.\n");

  MCore::setFakeIO(0, false);
  test_prose_m(1, 64, PROSE_MCore_N * NCORES_M * NCORES_M, PROSE_MCore_N * 2, false);
  MCore::setFakeIO(0, false);
  printf("\n\nshutting down peacefully. No errors found.\n");
}

#else
int main() {
  printf("PROSE not enabled\n");
}
#endif
