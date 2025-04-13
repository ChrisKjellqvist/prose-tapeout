//
// Created by Christopher Kjellqvist on 2/10/24.
//

#ifndef C_EXEC_UTIL_H
#define C_EXEC_UTIL_H
//
// Created by Christopher Kjellqvist on 1/16/24.
//
#ifndef BAREMETAL
#include <cstdint>
#include <cstdio>
#include <functional>
#include <optional>
#include <string>
#include <variant>

#else
#include <cstdint>
#endif

std::string get_global_checkpoint_dir();
std::string get_text_checkpoint_dir();

void matmul(const float* a, const float* b, float* c, int local_M, int local_Q, int local_K, float local_scalar_mult);

void print_matrix(float* f, int r, int c);
void print_matrix(uint16_t* f, int r, int c);
// traditional matrix multiply

// input is a m x n matrix
void convertRowMajorFormatToProSEColMajor(const uint16_t* in, uint16_t* out, int local_m, int local_n, int batch_size,
                                          int TILE_DIM);
// input is M x N matrix
void convertRowMajorFormatToProSERowMajor(const uint16_t* in, uint16_t* out, int local_M, int local_N, int batch_size,
                                          int TILE_WIDTH);

void convertPCMtoTCM(const uint16_t* in, std::variant<uint16_t*, float*> out, int local_M, int local_N, int batch_size,
                     int TILE_WIDTH);
void convertPRMtoTRM(const uint16_t* in, std::variant<uint16_t*, float*> out, int local_M, int local_N, int batch_size,
                     int TILE_WIDTH);
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
void write_to_file(const char* filename, const std::vector<std::pair<uint16_t*, int>>& data,
                   std::optional<std::pair<std::string, std::vector<std::string>>> index_writeout);
#endif

void memcpy_fp32_to_bf16(uint16_t* dst, float* src, size_t n_floats);

void memcpy_bf16_to_fp32(float* dst, uint16_t* src, size_t n_floats);

#endif // C_EXEC_UTIL_H
