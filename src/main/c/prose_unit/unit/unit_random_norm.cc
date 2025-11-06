//
// Created by Christopher Kjellqvist on 7/11/24.
//
#include <beethoven_hardware.h>
#include <beethoven/fpga_handle.h>
#include <algorithm>
#include <cmath>
#include <random>

using namespace beethoven;

const int N = 32;
const int batch_size = 2;
const int NORM_TO_TEST = flagLayerNorm;

int main() {

  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<float> dist(-5, 5);

  fpga_handle_t handle;
  auto set = handle.malloc(2 * N * batch_size);
  auto gb = handle.malloc(2 * N * 2);
  auto output = handle.malloc(2 * N * batch_size);

  auto input_ptr = (uint16_t *) set.getHostAddr();
  auto gb_ptr = (uint32_t *) gb.getHostAddr();
  auto output_ptr = (uint16_t *) output.getHostAddr();

  float golden[N][batch_size];
  float golden_input[N][batch_size];
  float golden_g[N];
  float golden_b[N];


  for (int i = 0; i < N; ++i) {
    for (int as = 0; as < batch_size; ++as) {
      auto j = (float) dist(gen);
      uint32_t j_cast = reinterpret_cast<uint32_t &>(j);
      input_ptr[i * batch_size + as] = uint16_t(j_cast >> 16);
      golden_input[i][as] = j;
    }
    auto g = 1 + (float) i / 2;
    auto b = (float) 0;
    uint32_t gb_fuse = (reinterpret_cast<uint32_t &>(g) & 0xFFFF0000L) | (reinterpret_cast<uint32_t &>(b) >> 16);
    gb_ptr[i] = gb_fuse;

    golden_g[i] = g;
    golden_b[i] = b;
  }

  // get golden model
  float mean[batch_size] = {0};
  for (int as = 0; as < batch_size; ++as) {
    for (int i = 0; i < N; ++i) {
      mean[as] += golden_input[i][as];
    }
    mean[as] /= N;
  }

  float var[batch_size] = {0};
  for (int as = 0; as < batch_size; ++as) {
    for (int i = 0; i < N; ++i) {
      float diff = golden_input[i][as] - mean[as];
      var[as] += diff * diff;
      printf("var[%d][%d]: %0.4f\n", as, i, var[as]);
    }
    printf("golden acc: %0.4f\n", var[as]);
    var[as] /= N;
    float lut_in = var[as] + 1e-5;
    float inv_std = 1.0 / std::sqrt(lut_in);

    for (int i = 0; i < N; ++i) {
      golden[i][as] = (golden_input[i][as] - mean[as]) * inv_std * golden_g[i] + golden_b[i];
      printf("golden gvar[%d]: %0.4f\n", i, golden_g[i] * inv_std);
      printf("golden diff[%d]: %0.4f\n", i, golden[i][as] - golden_input[i][as]);

    }
    printf("golden luti: %0.4f\n", lut_in);
    printf("golden mean: %0.4f\n", mean[as]);
    printf("golden invv: %0.4f\n", inv_std);
  }

  const float norm_f = 1.0f / N;
  const uint16_t norm = reinterpret_cast<const uint32_t &>(norm_f) >> 16;

  Norm::norm(0, gb, set, batch_size, 1, norm, NORM_TO_TEST, output, 1, N).get();

  float p1, p2;
  uint16_t hexdiff = 0;

#define unsigned_diff(x, y) ((x) > (y) ? (x) - (y) : (y) - (x))

  std::cout << "OUTPUT: output, golden" << std::endl;
  for (int as = 0; as < batch_size; ++as) {
    for (int i = 0; i < N; ++i) {
      uint16_t out = output_ptr[i * batch_size + as];
      uint32_t out_extends = uint32_t(out) << 16;
      float out_f = reinterpret_cast<float &>(out_extends);
      uint16_t gold_hex = reinterpret_cast<uint32_t &>(golden[i][as]) >> 16;
      printf("%04x, %04x\t%0.4f, %0.4f\n", out, gold_hex, out_f, golden[i][as]);
      if (unsigned_diff(out, gold_hex) > hexdiff) {
        hexdiff = unsigned_diff(out, gold_hex);
        printf("(%04x - %04x) = %04x\n", out, gold_hex, hexdiff);
        p1 = out_f;
        p2 = golden[i][as];
      }
    }
  }

  printf("Max diff: %x\n", hexdiff);
  printf("Max diff: %0.4f, %0.4f\n", p1, p2);
}
