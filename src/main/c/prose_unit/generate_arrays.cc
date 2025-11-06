//
// Created by Christopher Kjellqvist on 3/27/24.
//

#include "../utility/util.h"
#include <random>
#include <cmath>

float activation_f(float q) {
  return exp(q);
}

int main() {
  int M = 4;
  int K = 32;
  int Q = 4;
  int chosen_batch_size = 1;
  int tile_size = 4;
  bool output_transpose = false;

  float *activation[chosen_batch_size];
  float weights[Q * K];
  float *golden_matmul[chosen_batch_size];
  float *golden_pa_matmul[chosen_batch_size];
  for (int i = 0; i < chosen_batch_size; ++i) activation[i] = new float[M * K];
  for (int i = 0; i < chosen_batch_size; ++i) golden_matmul[i] = new float[Q * M];
  for (int i = 0; i < chosen_batch_size; ++i) golden_pa_matmul[i] = new float[Q * M];

  float golden_norms[M * chosen_batch_size];

  auto prose_activation_transpose = new uint16_t[M * K * chosen_batch_size];
  auto prose_weights = new uint16_t[Q * K * chosen_batch_size];

  uint16_t bfloat_act[chosen_batch_size][M * K];
  uint16_t bfloat_wgt[Q * K];

  auto prose_host_act = prose_activation_transpose;
  auto prose_host_wgt = prose_weights;

  std::random_device rd;
  std::mt19937 eng(123); // rd());
  std::uniform_real_distribution<float> dist(-2, 2);

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
    convertRowMajorFormatToProSEColMajor(bfloat_act[b], prose_host_act,
                                         M, b, chosen_batch_size, tile_size);
  }

  // initialize weights
  for (int i = 0; i < K * Q; ++i) {
    auto wgt_raw = dist(eng);
    uint32_t wgt_hex = reinterpret_cast<uint32_t &>(wgt_raw);
    wgt_hex &= 0xffff0000;
    float wgt = reinterpret_cast<float &>(wgt_hex);
    weights[i] = wgt;
    bfloat_wgt[i] = uint16_t(wgt_hex >> 16);
  }
  convertRowMajorFormatToProSERowMajor(bfloat_wgt, prose_host_wgt, K, Q, 1, tile_size);

  for (int i = 0; i < chosen_batch_size; ++i) {
    matmul(activation[i], weights, golden_matmul[i], M, Q, K, 1);
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
  auto prose_host_res = new uint16_t[M * Q * chosen_batch_size];
  if (output_transpose) {
    convertRowMajorFormatToProSERowMajor(reinterpret_cast<uint16_t *>(golden_matmul[0]),
                                         prose_host_res, M, Q, 1, tile_size);
  } else {
    convertRowMajorFormatToProSERowMajor(reinterpret_cast<uint16_t *>(golden_matmul[0]),
                                         prose_host_res, M, Q, 1, tile_size);
  }
  // write out to file
//  write_to_file("arrays.hex",
//                {{prose_activation_transpose, M * K * chosen_batch_size},
//                               {prose_weights, Q * K * chosen_batch_size},
//                               {prose_host_res, M * Q * chosen_batch_size}});
throw std::exception();
}