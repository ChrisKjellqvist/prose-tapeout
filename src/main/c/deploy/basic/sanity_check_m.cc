#include "prose_lib.h"
#include <beethoven/fpga_handle.h>
#include <beethoven_hardware.h>
#include <cstdio>
#include <cstring>

int main() {
  uint16_t ZERO = 0;
  uint16_t ONE = 0x3f80;
  uint16_t CHOSEN = 0x4049; // pi
  int stripe_size = 2;

  beethoven::fpga_handle_t handle;

  int K = 8;
  int N = 8;
  int M = K;

  auto identity_matrix = handle.malloc(2 * M * K);
  auto host_ident = (uint16_t *)identity_matrix.getHostAddr();
  for (int r_s = 0; r_s < M / stripe_size; ++r_s) {
    for (int c = 0; c < K; ++c) {
      for (int rsi = 0; rsi < stripe_size; ++rsi) {
        int row = r_s * stripe_size + rsi;
        *(host_ident++) = (row == c) ? ONE : ZERO;
      }
    }
  }

  auto w_matrix = handle.malloc(2 * N * K);
  auto w_host = (uint16_t *)w_matrix.getHostAddr();
  for (int r = 0; r < N; ++r) {
    for (int c = 0; c < K; ++c) {
      *(w_host++) = CHOSEN;
    }
  }

  auto output = handle.malloc(2 * M * N);
  memset(output.getHostAddr(), 0, 2 * M * N);

  prose_m_matmul(identity_matrix, w_matrix, output, nullptr, PROSE_biasNONE, 1,
                 M, K, N, true, nullptr, false, false);

  auto host_out = (uint16_t*)output.getHostAddr();
  for (int r = 0; r < N; ++r) {
    for (int c = 0; c < K; ++c) {
      printf("%04x ", *(host_out++));
    }
    printf("\n");
  }
}