//
// Created by Christopher Kjellqvist on 2/10/24.
//

#ifndef C_EXEC_UTIL_H
#define C_EXEC_UTIL_H
//
// Created by Christopher Kjellqvist on 1/16/24.
//
#ifndef BAREMETAL
#include <cstdio>
#include <cstdint>
#include <functional>
#include <optional>
#include <string>
#include <variant>
#ifdef USE_TORCH
#include <torch/torch.h>
#endif

#else
#include <cstdint>
#endif

void matmul(const float *a, const float *b, float *c, int local_M, int local_Q, int local_K, float local_scalar_mult);

void print_matrix(float *f, int r, int c);
void print_matrix(uint16_t *f, int r, int c);
// traditional matrix multiply

// input is a m x n matrix
void
convertRowMajorFormatToProSEColMajor(const uint16_t *in, uint16_t *out,
                                     int local_m, int local_n,
                                     int batch_size, int TILE_DIM);
// input is M x N matrix
void convertRowMajorFormatToProSERowMajor(const uint16_t *in, uint16_t *out,
                                          int local_M, int local_N,
                                          int batch_size, int TILE_WIDTH);

void convertPCMtoTCM(const uint16_t *in, std::variant<uint16_t *, float *> out,
                     int local_M, int local_N,
                     int batch_size, int TILE_WIDTH);
void convertPRMtoTRM(const uint16_t *in, std::variant<uint16_t*, float*> out,
                     int local_M, int local_N,
                     int batch_size, int TILE_WIDTH);
float bf16_to_float(uint16_t q);

// https://pytorch.org/docs/stable/generated/torch.nn.GELU.html
float gelu(float x);
float truncate_float_to_bf16_accuracy(float q);
bool f_is_zero(float q);
bool bit_accurate_is_equal(float q, float r);
bool is_approximately_infinity(float q);

bool generous_is_equal(float q, float r);

uint16_t float_to_bf16(float q);

void reset_error_measurement();
bool super_permissive_equals(float q, float r, float max_err_percentage, int idx);
void print_max_err();
#ifndef BAREMETAL
void write_to_file(const std::string &filename_prefix,
                   const std::vector<std::pair<uint16_t*, int>> &data,
                   std::optional<std::pair<std::string, std::vector<std::string>>> index_writeout);
#endif
#ifdef USE_TORCH
torch::Tensor load_tensor(const std::string &filename);

void print_small_tensor(torch::Tensor tensor);

void print_small_tensor_1d(torch::Tensor tensor);

void print_2d_tensor(torch::Tensor tensor);

void print_batched_2d_tensor(const torch::Tensor &tensor);

void print_1d_tensor(torch::Tensor tensor);

void print_1d_tensor_as_16hex(torch::Tensor tensor);

float max_pct_diff(const torch::Tensor &a, const torch::Tensor &b);

torch::Tensor random_tensor(int batch_size, int seq_len, int embed_dim);

torch::Tensor random_tensor(int M, int N);

torch::Tensor random_tensor(int dim);
#endif

#endif //C_EXEC_UTIL_H
