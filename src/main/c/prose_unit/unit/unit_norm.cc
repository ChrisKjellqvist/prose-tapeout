//
// Created by Christopher Kjellqvist on 7/11/24.
//
#include <beethoven_hardware.h>
#include <beethoven/fpga_handle.h>
#include <cmath>
#include <algorithm>

using namespace beethoven;

const int N = 32;
const int batch_size = 4;

const int NORM_TO_TEST = flagRMSNorm;

float **golden_layernorm(float **x, float *gamma, float *beta, size_t n, size_t batch_size) {
  float **all_x = new float *[batch_size];
  for (int b = 0; b < batch_size; ++b) {
    float *y = new float[n];
    float mean = 0;
    for (int i = 0; i < n; ++i) {
      mean += x[b][i];
    }
    mean /= (float) n;

    float var = 0;
    for (int i = 0; i < n; ++i) {
      float diff = x[b][i] - mean;
      var += diff * diff;
    }
    var /= (float) n;
    float inv_std = 1.0 / std::sqrt(var + 1e-5);

    for (int i = 0; i < n; ++i) {
      y[i] = (x[b][i] - mean) * inv_std * gamma[i] + beta[i];
    }
    all_x[b] = y;
  }

  return all_x;
}

float **golden_rmsnorm(float **x, float *gamma, size_t n, size_t batch_size) {
  float **all_x = new float *[batch_size];
  for (int b = 0; b < batch_size; ++b) {
    float *y = new float[n];
    float mean = 0;
    for (int i = 0; i < n; ++i) {
      mean += x[b][i] * x[b][i];
    }
    mean /= (float) n;

    float inv_std = 1.0 / std::sqrt(mean + 1e-5);

    printf("norm: %f\n", inv_std);

    for (int i = 0; i < n; ++i) {
      y[i] = x[b][i] * inv_std * gamma[i];
    }
    all_x[b] = y;
  }
  return all_x;
}

int main() {
  fpga_handle_t handle;
  auto set = handle.malloc(2 * N * batch_size);
  auto gb = handle.malloc(2 * N * 2);
  auto output = handle.malloc(2 * N * batch_size);

  auto input_ptr = (uint16_t *) set.getHostAddr();
  auto gb_ptr = (uint32_t *) gb.getHostAddr();
  auto output_ptr = (uint16_t *) output.getHostAddr();

  float **golden;
  float **golden_input = new float *[batch_size];
  float golden_g[N];
  float golden_b[N];

  for (int as = 0; as < batch_size; ++as) {
    golden_input[as] = new float[N];
    for (int i = 0; i < N; ++i) {
      auto j = (float) i * (as + 1);
      uint32_t j_cast = reinterpret_cast<uint32_t &>(j);
      input_ptr[i * batch_size + as] = uint16_t(j_cast >> 16);
      golden_input[as][i] = j;

      // only needs to be initialized once
      if (as == 0) {
        auto g = 1 + (float) i / 2;
        auto b = (float) 0;
        uint32_t gb_fuse = (reinterpret_cast<uint32_t &>(g) & 0xFFFF0000L) | (reinterpret_cast<uint32_t &>(b) >> 16);
        gb_ptr[i] = gb_fuse;

        golden_g[i] = g;
        golden_b[i] = b;
      }
    }
  }

  switch (NORM_TO_TEST) {
    case flagLayerNorm:
      golden = golden_layernorm(golden_input, golden_g, golden_b, N, batch_size);
      break;
    case flagRMSNorm:
      golden = golden_rmsnorm(golden_input, golden_g, N, batch_size);
      break;
    default:
      throw std::runtime_error("Unknown norm type");
  }

  const float norm_f = 1.0f / N;
  const uint16_t norm = reinterpret_cast<const uint32_t &>(norm_f) >> 16;

  Norm::norm(0, gb, set, batch_size, 1, norm, NORM_TO_TEST, output, 1, N).get();

  std::cout << "OUTPUT: output, golden" << std::endl;
  for (int as = 0; as < batch_size; ++as) {
    for (int i = 0; i < N; ++i) {
      uint16_t out = output_ptr[i * batch_size + as];
      uint32_t out_extends = uint32_t(out) << 16;
      float out_f = reinterpret_cast<float &>(out_extends);
      uint16_t gold_hex = reinterpret_cast<uint32_t &>(golden[as][i]) >> 16;
      printf("%04x, %04x\t%0.4f, %0.4f\n", out, gold_hex, out_f, golden[as][i]);
    }
  }

  handle.shutdown();
}
