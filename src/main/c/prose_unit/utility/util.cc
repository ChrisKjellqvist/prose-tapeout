//
// Created by Christopher Kjellqvist on 1/16/24.
//

#include "util.h"
#include <cinttypes>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <functional>
#include <iostream>

void print_matrix(float* f, int r, int c) {
  const int print_column_width = 12;
  for (int i = 0; i < r; ++i) {
    for (int j = 0; j < c; ++j) {
      uint32_t q = reinterpret_cast<uint32_t&>(f[i * c + j]);
      printf("%-*f(%04x) ", print_column_width, f[i * c + j], q >> 16);
    }
    printf("\n");
  }
}

void print_matrix(uint16_t* f, int r, int c) {
  for (int i = 0; i < r; ++i) {
    for (int j = 0; j < c; ++j) {
      printf("%04x ", f[i * c + j]);
    }
    printf("\n");
  }
}

// traditional matrix multiply
void matmul(const float* a, const float* b, float* c, int local_M, int local_Q,
            int local_K, float local_scalar_mult) {
  for (int i = 0; i < local_M; ++i) {
    for (int j = 0; j < local_Q; ++j) {
      float acc = 0;
      for (int k = 0; k < local_K; ++k) {
        acc += truncate_float_to_bf16_accuracy(a[i * local_K + k]) *
            truncate_float_to_bf16_accuracy(b[k * local_Q + j]);
      }
      c[i * local_Q + j] = acc * local_scalar_mult;
    }
  }
}

// input is a m x n matrix
void convertRowMajorFormatToProSEColMajor(const uint16_t* in, uint16_t* out,
                                          int local_m, int local_n,
                                          int batch_size, int TILE_WIDTH) {
  for (int b = 0; b < batch_size; ++b) {
    for (int row = 0; row < local_m; ++row) {
      int stripe = row / TILE_WIDTH;
      int stripe_subIdx = row % TILE_WIDTH;
      auto base = stripe * local_n * TILE_WIDTH * batch_size;
      for (int col = 0; col < local_n; ++col) {
        out[base + col * TILE_WIDTH * batch_size + b * TILE_WIDTH +
            stripe_subIdx] = in[b * local_n * local_m + row * local_n + col];
      }
    }
  }
}

float bf16_to_float(uint16_t q) {
  uint32_t q_t = (uint32_t(q)) << 16;
  return reinterpret_cast<float&>(q_t);
}

// input is M x N matrix
void convertRowMajorFormatToProSERowMajor(const uint16_t* in, uint16_t* out,
                                          int local_M, int local_N,
                                          int batch_size, int TILE_WIDTH) {
  for (int b = 0; b < batch_size; ++b) {
    for (int col = 0; col < local_N; ++col) {
      int stripe = col / TILE_WIDTH;
      int stripe_subIdx = col % TILE_WIDTH;
      auto stripe_sz_ele = TILE_WIDTH * local_M * batch_size;
      for (int row = 0; row < local_M; ++row) {
        out[stripe * stripe_sz_ele + row * TILE_WIDTH * batch_size +
            b * TILE_WIDTH + +stripe_subIdx] =
            in[b * local_M * local_N + row * local_N + col];
      }
    }
  }
}

void convertPRMtoTRM(const uint16_t* in, std::variant<uint16_t*, float*> out,
                     int local_M, int local_N, int batch_size, int TILE_WIDTH) {
  for (int b = 0; b < batch_size; ++b) {
    for (int col = 0; col < local_N; ++col) {
      int stripe = col / TILE_WIDTH;
      int stripe_subIdx = col % TILE_WIDTH;
      auto stripe_sz_ele = TILE_WIDTH * local_M * batch_size;
      for (int row = 0; row < local_M; ++row) {
        uint16_t ele =
            in[stripe * stripe_sz_ele + row * TILE_WIDTH * batch_size +
               b * TILE_WIDTH + +stripe_subIdx];
        if (std::holds_alternative<float*>(out)) {
          std::get<float*>(out)[b * local_N * local_M + row * local_N + col] =
              bf16_to_float(ele);
        } else {
          std::get<uint16_t*>(
              out)[b * local_N * local_M + row * local_N + col] = ele;
        }
      }
    }
  }
}


void convertPCMtoTCM(const uint16_t* in, std::variant<uint16_t*, float*> out,
                     int local_M, int local_N, int batch_size, int TILE_WIDTH) {
  for (int b = 0; b < batch_size; ++b) {
    for (int row = 0; row < local_M; ++row) {
      int stripe = row / TILE_WIDTH;
      int stripe_subIdx = row % TILE_WIDTH;
      auto base = stripe * local_N * TILE_WIDTH * batch_size;
      for (int col = 0; col < local_N; ++col) {
        auto ele = in[base + col * TILE_WIDTH * batch_size + b * TILE_WIDTH +
                      stripe_subIdx];
        if (std::holds_alternative<float*>(out)) {
          std::get<float*>(out)[b * local_N * local_M + row * local_N + col] =
              bf16_to_float(ele);
        } else {
          std::get<uint16_t*>(
              out)[b * local_N * local_M + row * local_N + col] = ele;
        }
      }
    }
  }
}


#define IS_BIGENDIAN (*(uint16_t*)"\0\xff" < 0x100)

#include <optional>
#include <string>

// https://pytorch.org/docs/stable/generated/torch.nn.GELU.html
float gelu(float x) { return float((x * 0.5) * (1 + std::erf(x / M_SQRT2))); }

float truncate_float_to_bf16_accuracy(float q) {
  uint32_t q_t = reinterpret_cast<uint32_t&>(q);
  uint32_t masked = q_t & 0xFFFF0000;
  if (q_t & 0x8000) {
    masked += 0x10000;
  }
  return reinterpret_cast<float&>(masked);
}

bool f_is_zero(float q) {
  return (reinterpret_cast<uint32_t&>(q) & 0x7FFF0000) == 0;
}

bool bit_accurate_is_equal(float q, float r) {
  return q == r || (f_is_zero(q) && f_is_zero(r));
}

/**
 * The original prose paper proposes a lookup table that is a piecewise
 * approximation of GeLU / Exp. Of course, yes, the C++ baseline implementation
 * could implement such a lookup table but that doesn't make a whole lot of
 * sense when the point is to make sure that the lookup table aligns with
 * software results.
 *
 * Instead, for the C++ baseline, we use the actual GeLU and Exp functions
 * and then use an approximate equality function, shown below. Bit-accurate
 * equality can be tested with the above function.
 */
bool is_approximately_infinity(float q) { return std::abs(q) > 1e10; }

bool generous_is_equal(float q, float r) {
  return q == r || std::abs(q - r) < 0.0001 ||
      (is_approximately_infinity(q) && is_approximately_infinity(r));
}

uint16_t float_to_bf16(float q) {
  uint32_t a = reinterpret_cast<uint32_t&>(q);
  uint16_t b = a >> 16;
  if (a & 0x8000)
    b += 1;
  return b;
}

#include <tuple>
#include <vector>

std::vector<float> max_error;
std::vector<std::pair<float, float>> err_pairs;

void reset_error_measurement() {
  max_error.clear();
  err_pairs.clear();
}

bool super_permissive_equals(float q, float r, float max_err_percentage,
                             int batch) {
  if (max_error.size() <= batch) {
    auto pre_size = max_error.size();
    max_error.resize(batch + 1);
    err_pairs.resize(batch + 1);
    for (auto i = pre_size; i < max_error.size(); ++i) {
      max_error[i] = 0;
    }
  }
  float err = std::min(std::abs(1 - (q / r)), std::abs(q - r));
  bool eq = q == r || err < max_err_percentage ||
      (is_approximately_infinity(q) && is_approximately_infinity(r));
  if (err > max_error[batch] && !is_approximately_infinity(q)) {
    max_error[batch] = err;
    err_pairs[batch] = std::make_pair(q, r);
  }
  if (!eq) {
    printf("Error: %0.4f%%\n", err);
    fflush(stdout);
  }
  return eq;
}

void print_max_err() {
  for (int i = 0; i < max_error.size(); ++i) {
    printf("For batch %d:\n", i);
    if (max_error[i] > 0)
      printf("Maximum error (%0.4f%%) was produced by qr pair q%0.4f r%0.4f\n",
             max_error[i], err_pairs[i].first, err_pairs[i].second);
    else
      printf("No error found\n");
  }
  fflush(stdout);
}

void memcpy_bf16_to_fp32(float* dst, uint16_t* src, size_t n_floats) {
  for (int i = 0; i < n_floats; ++i) {
    dst[i] = bf16_to_float(src[i]);
  }
}

void memcpy_fp32_to_bf16(uint16_t* dst, float* src, size_t n_floats) {
  for (int i = 0; i < n_floats; ++i) {
    dst[i] = float_to_bf16(src[i]);
  }
}

// https://stackoverflow.com/questions/478898/how-do-i-execute-a-command-and-get-the-output-of-the-command-within-c-using-po
std::string exec(const char* cmd) {
  char buffer[128];
  std::string result = "";
  FILE* pipe = popen(cmd, "r");
  if (!pipe)
    throw std::runtime_error("popen() failed!");
  try {
    while (fgets(buffer, sizeof buffer, pipe) != NULL) {
      result += buffer;
    }
  } catch (...) {
    pclose(pipe);
    throw;
  }
  pclose(pipe);
  return result;
}


// // chris work computer
////const std::string GLOBAL_CHECKPOINT_DIRECTORY =
////
///"/Users/chriskjellqvist/Code/prose/prose_rtl/c_exec/src/prose_unit/neo_test/gen_ckpts/model_ckpts";
////const std::string TEXT_CHECKPOINT_DIRECTORY =
////
///"/Users/chriskjellqvist/Code/prose/prose_rtl/c_exec/src/prose_unit/neo_test/gen_ckpts/txt_ckpts";
//
//// chris home computer
////const std::string GLOBAL_CHECKPOINT_DIRECTORY =
////
///"/Users/chris/Code/prose_rtl/c_exec/src/prose_unit/neo_test/gen_ckpts/model_ckpts";
////const std::string TEXT_CHECKPOINT_DIRECTORY =
////
///"/Users/chris/Code/prose_rtl/c_exec/src/prose_unit/neo_test/gen_ckpts/txt_ckpts";
//
//
//// oak
// const std::string GLOBAL_CHECKPOINT_DIRECTORY =
// "/home/chriskjellqvist/Code/prose_rtl/c_exec/src/prose_unit/neo_test/gen_ckpts/model_ckpts";
// const std::string TEXT_CHECKPOINT_DIRECTORY =
// "/home/chriskjellqvist/Code/prose_rtl/c_exec/src/prose_unit/neo_test/gen_ckpts/txt_ckpts";
//
//// kria
// const std::string REMOTE_TXT_DIRECTORY = "/home/petalinux/prose_ins";

std::string get_global_checkpoint_dir() {
  auto hostname = exec("hostname");
  hostname = hostname.substr(0, hostname.size() - 1);
  if (hostname == "Christophers-MacBook-Air-2.local") {
    return "/Users/chris/Code/prose_rtl/c_exec/src/prose_unit/neo_test/"
           "gen_ckpts/model_ckpts";
  } else if (hostname == "oak") {
    return "/home/chriskjellqvist/Code/prose_rtl/c_exec/src/prose_unit/"
           "neo_test/gen_ckpts/model_ckpts";
  } else {
    throw std::runtime_error("couldn't match hostname '" + hostname + "'");
  }
}

std::string get_text_checkpoint_dir() {
  auto hostname = exec("hostname");
  // get rid of newline
  hostname = hostname.substr(0, hostname.size() - 1);
  if (hostname == "Christophers-MacBook-Air-2.local") {
    return "/Users/chris/Code/prose_rtl/c_exec/src/prose_unit/neo_test/"
           "gen_ckpts/txt_ckpts";
  } else if (hostname == "oak") {
    return "/home/chriskjellqvist/Code/prose_rtl/c_exec/src/prose_unit/"
           "neo_test/gen_ckpts/txt_ckpts";
  } else if (hostname == "xilinx-kv260-starterkit-20241") {
    return "/home/petalinux/prose_ins";
  } else {
    throw std::runtime_error("couldn't find checkpoint");
  }
}
