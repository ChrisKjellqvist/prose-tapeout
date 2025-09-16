//
// Created by Christopher Kjellqvist on 1/16/24.
//

#include <cmath>
#include <cstdio>
#include <functional>
#include <tuple>
#include <iostream>
#include <cinttypes>
#include <fstream>
#include "util.h"

#include "torch/torch.h"

void print_matrix(float *f, int r, int c) {
  const int print_column_width = 12;
  for (int i = 0; i < r; ++i) {
    for (int j = 0; j < c; ++j) {
      uint32_t q = reinterpret_cast<uint32_t &>(f[i * c + j]);
      printf("(%04xd)%-*f ", print_column_width, q >> 16, f[i * c + j]);
    }
    printf("\n");
  }
}

void print_matrix(uint16_t *f, int r, int c
) {
  for (
          int i = 0;
          i < r;
          ++i) {
    for (
            int j = 0;
            j < c;
            ++j) {
      printf("%04x ", f[
              i * c
              + j]);
    }
    printf("\n");
  }
}

// traditional matrix multiply
void matmul(const float *a, const float *b, float *c, int local_M, int local_Q, int local_K, float local_scalar_mult) {
  for (int i = 0; i < local_M; ++i) {
    for (int j = 0; j < local_Q; ++j) {
      float acc = 0;
      for (int k = 0; k < local_K; ++k) {
        acc += a[i * local_K + k] * b[k * local_Q + j];
      }
      c[i * local_Q + j] = acc * local_scalar_mult;
    }
  }
}

// input is a m x n matrix
void convertRowMajorFormatToProSEColMajor(const uint16_t *in, uint16_t *out,
                                          int local_m, int local_n,
                                          int batch_size,
                                          int TILE_WIDTH) {
  for (int b = 0; b < batch_size; ++b) {
    for (int row = 0; row < local_m; ++row) {
      int stripe = row / TILE_WIDTH;
      int stripe_subIdx = row % TILE_WIDTH;
      auto base = stripe * local_n * TILE_WIDTH * batch_size;
      for (int col = 0; col < local_n; ++col) {
        out[base +
            col * TILE_WIDTH * batch_size +
            b * TILE_WIDTH +
            stripe_subIdx] = in[b * local_n * local_m + row * local_n + col];
      }
    }
  }
}

float bf16_to_float(uint16_t q) {
  uint32_t q_t = (uint32_t(q)) << 16;
  return reinterpret_cast<float &>(q_t);
}

// input is M x N matrix
void convertRowMajorFormatToProSERowMajor(const uint16_t *in, uint16_t *out,
                                          int local_M, int local_N,
                                          int batch_size, int TILE_WIDTH) {
  for (int b = 0; b < batch_size; ++b) {
    for (int col = 0; col < local_N; ++col) {
      int stripe = col / TILE_WIDTH;
      int stripe_subIdx = col % TILE_WIDTH;
      auto stripe_sz_ele = TILE_WIDTH * local_M * batch_size;
      for (int row = 0; row < local_M; ++row) {
        out[stripe * stripe_sz_ele +
            row * TILE_WIDTH * batch_size +
            b * TILE_WIDTH +
            +stripe_subIdx] = in[b * local_M * local_N + row * local_N + col];
      }
    }
  }
}

void convertPRMtoTRM(const uint16_t *in, std::variant<uint16_t *, float *> out,
                     int local_M, int local_N,
                     int batch_size, int TILE_WIDTH) {
  for (int b = 0; b < batch_size; ++b) {
    for (int col = 0; col < local_N; ++col) {
      int stripe = col / TILE_WIDTH;
      int stripe_subIdx = col % TILE_WIDTH;
      auto stripe_sz_ele = TILE_WIDTH * local_M * batch_size;
      for (int row = 0; row < local_M; ++row) {
        uint16_t ele = in[stripe * stripe_sz_ele +
                          row * TILE_WIDTH * batch_size +
                          b * TILE_WIDTH +
                          +stripe_subIdx];
        if (std::holds_alternative<float *>(out)) {
          std::get<float *>(out)[b * local_N * local_M + row * local_N + col] = bf16_to_float(ele);
        } else {
          std::get<uint16_t *>(out)[b * local_N * local_M + row * local_N + col] = ele;
        }
      }
    }
  }
}


void convertPCMtoTCM(const uint16_t *in, std::variant<uint16_t *, float *> out,
                     int local_M, int local_N,
                     int batch_size, int TILE_WIDTH) {
  for (int b = 0; b < batch_size; ++b) {
    for (int row = 0; row < local_M; ++row) {
      int stripe = row / TILE_WIDTH;
      int stripe_subIdx = row % TILE_WIDTH;
      auto base = stripe * local_N * TILE_WIDTH * batch_size;
      for (int col = 0; col < local_N; ++col) {
        auto ele = in[base +
                      col * TILE_WIDTH * batch_size +
                      b * TILE_WIDTH +
                      stripe_subIdx];
        if (std::holds_alternative<float *>(out)) {
          std::get<float *>(out)[b * local_N * local_M + row * local_N + col] = bf16_to_float(ele);
        } else {
          std::get<uint16_t *>(out)[b * local_N * local_M + row * local_N + col] = ele;
        }
      }
    }
  }
}


#define IS_BIGENDIAN (*(uint16_t *) "\0\xff" < 0x100)

#include <optional>
#include <string>

#ifndef BAREMETAL

const int seq_len_for_mask = 16;

void write_to_file(const char *filename,
                   const std::vector<std::pair<uint16_t *, int>> &data,
                   std::optional<std::pair<std::string, std::vector<std::string>>> index_writeout) {
  FILE *f = fopen(filename, "wb");
  std::vector<uintptr_t> index;
  if (!f) {
    printf("Failed to open file %s\n", filename);
    return;
  }
  uintptr_t addr = 0;
  // always write out data in bigendian format
  for (auto &d: data) {
    auto dptr = d.first;
    index.push_back(addr);
    if (IS_BIGENDIAN) {
      auto swapped = new uint16_t[d.second];
      for (int i = 0; i < d.second; ++i) {
        swapped[i] = (d.first[i] >> 8) | (d.first[i] << 8);
      }
      dptr = swapped;
    }
    fprintf(f, "@%lx\n", addr);
    for (int i = 0; i < d.second; ++i) {
      for (int j = 0; j < 2; ++j)
        fprintf(f, "%02x ", 0xFF & (dptr[i] >> (8 * j)));
      if (i % 8 == 7) fprintf(f, "\n");
    }

    if (IS_BIGENDIAN) {
      delete[] dptr;
    }
    fprintf(f, "\n");
    addr += d.second * 2;
    if (addr % 4096 != 0) {
      addr += 4096 - (addr % 4096);
    }
  }
  fclose(f);

  if (index_writeout) {
    FILE *f = fopen(index_writeout.value().first.c_str(), "w");
    FILE *rp = fopen("prose_rptr.h", "w");
    fprintf(rp, "#ifndef PROSE_RPTR_H\n#define PROSE_RPTR_H\n");
    fprintf(rp, "#include <cstdint>\n");
    fprintf(rp, "#include <beethoven_baremetal/allocator/alloc_baremetal.h>\n");
    if (!f) {
      printf("Failed to open file %s\n", index_writeout.value().first.c_str());
      return;
    }
    for (int i = 0; i < index.size(); ++i) {
      fprintf(f, "%s %lx\n", index_writeout.value().second[i].c_str(), index[i]);
      // replace all "." in name with "_"
      std::string name = index_writeout.value().second[i];
      std::replace(name.begin(), name.end(), '.', '_');
      fprintf(rp, "constexpr beethoven::remote_ptr %s(0x%lxL);\n", name.c_str(), index[i]);
    }
    fprintf(rp, "constexpr uint32_t allocator_base(0x%lx);\n", addr);
    fprintf(rp, "#endif\n");
    fprintf(f, "END: %lx\n", addr);
    fclose(f);
    fclose(rp);
  }
}

#endif

// https://pytorch.org/docs/stable/generated/torch.nn.GELU.html
float gelu(float x) {
  return float((x * 0.5) * (1 + std::erf(x / M_SQRT2)));
}

float truncate_float_to_bf16_accuracy(float q) {
  uint32_t q_t = reinterpret_cast<uint32_t &>(q);
  uint32_t masked = q_t & 0xFFFF0000;
  if (q_t & 0x8000) {
    masked += 0x10000;
  }
  return reinterpret_cast<float &>(masked);
}

bool f_is_zero(float q) {
  return (reinterpret_cast<uint32_t &>(q) & 0x7FFF0000) == 0;
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
bool is_approximately_infinity(float q) {
  return std::abs(q) > 1e10;
}

bool generous_is_equal(float q, float r) {
  return q == r || std::abs(q - r) < 0.0001 || (is_approximately_infinity(q) && is_approximately_infinity(r));
}

uint16_t float_to_bf16(float q) {
  uint32_t a = reinterpret_cast<uint32_t &>(q);
  uint16_t b = a >> 16;
  if (a & 0x8000) b += 1;
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

bool super_permissive_equals(float q, float r, float max_err_percentage, int batch) {
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

torch::Tensor load_tensor(const std::string &filename) {
  std::ifstream file(filename, std::ios::binary);
  std::vector<char> data((std::istreambuf_iterator<char>(file)),
                         std::istreambuf_iterator<char>());
  auto ivalue = torch::pickle_load(data);
  // get the tensor from the ivalue
  auto tensor = ivalue.toTensor();
  std::cout << "Loaded tensor" << filename << " with shape: " << tensor.sizes() << std::endl
            << std::flush;
  return tensor;
}


torch::Tensor random_tensor(int batch_size, int M, int N) {
  auto options = torch::TensorOptions().dtype(torch::kFloat32);
  return torch::randn({batch_size, M, N}, options).contiguous();
}

torch::Tensor random_tensor(int M, int N) {
  auto options = torch::TensorOptions().dtype(torch::kFloat32);
  return torch::randn({M, N}, options).contiguous();
}

torch::Tensor random_tensor(int dim) {
  auto options = torch::TensorOptions().dtype(torch::kFloat32);
  return torch::randn({dim}, options).contiguous();
}

void print_small_tensor(torch::Tensor tensor) {
  // Print the first 10x10 elements of the output tensor
  int rows_to_print = std::min(10, (int) tensor.size(0));
  int cols_to_print = std::min(10, (int) tensor.size(1));

  std::cout << "First 10x10 elements of the output tensor:" << std::endl;
  for (int i = 0; i < rows_to_print; ++i) {
    for (int j = 0; j < cols_to_print; ++j) {
      std::cout << tensor[i][j].item<float>() << " ";
    }
    std::cout << std::endl;
  }
}

void print_small_tensor_1d(torch::Tensor tensor) {
  // Print the first 10x10 elements of the output tensor
  int rows_to_print = std::min(10, (int) tensor.size(0));

  std::cout << "First 10x10 elements of the output tensor:" << std::endl;
  for (int i = 0; i < rows_to_print; ++i) {
    std::cout << tensor[i].item<float>() << " ";
  }
  std::cout << std::endl;
}

void print_2d_tensor(torch::Tensor tensor) {
  // Print the first 10x10 elements of the output tensor
  int rows_to_print = (int) tensor.size(0);
  int cols_to_print = (int) tensor.size(1);
  for (int i = 0; i < rows_to_print; ++i) {
    for (int j = 0; j < cols_to_print; ++j) {
      std::cout << tensor[i][j].item<float>() << " ";
    }
    std::cout << std::endl;
  }
}

void print_batched_2d_tensor(const torch::Tensor &tensor) {
  // Print the first 10x10 elements of the output tensor
  int batch_size = (int) tensor.size(0);
  int rows_to_print = (int) tensor.size(1);
  int cols_to_print = (int) tensor.size(2);
  for (int b = 0; b < batch_size; ++b) {
    std::cout << "Batch " << b << std::endl;
    for (int i = 0; i < rows_to_print; ++i) {
      for (int j = 0; j < cols_to_print; ++j) {
        std::cout << tensor[b][i][j].item<float>() << " ";
      }
      std::cout << std::endl;
    }
  }
}

void print_1d_tensor(torch::Tensor tensor) {
  // Print the first 10x10 elements of the output tensor
  int cols_to_print = (int) tensor.size(0);
  for (int i = 0; i < cols_to_print; ++i) {
    std::cout << tensor[i].item<float>() << " ";
  }
  std::cout << std::endl;
}

void print_1d_tensor_as_16hex(torch::Tensor tensor) {
  // Print the first 10x10 elements of the output tensor
  int cols_to_print = (int) tensor.size(0);
  for (int i = 0; i < cols_to_print; ++i) {
    float x = tensor[i].item<float>();
    uint32_t cast = reinterpret_cast<uint32_t &>(x);
    printf("%04x ", (uint16_t) (cast >> 16));
  }
  std::cout << std::endl;
}

float max_pct_diff(const torch::Tensor &a, const torch::Tensor &b) {
  auto diff = (a - b).abs();
  auto max_diff = diff.max().item<float>();
  auto max_a = a.abs().max().item<float>();
  auto max_b = b.abs().max().item<float>();
  return max_diff / std::max(max_a, max_b) * 100;
}