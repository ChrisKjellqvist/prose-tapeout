//
// Created by Chris Kjellqvist on 1/4/25.
//

#ifndef INTERCHANGE_FORMAT_H
#define INTERCHANGE_FORMAT_H

#include <string>
#include <vector>
#include <cinttypes>

#ifdef USE_TORCH
#include <torch/torch.h>
#endif

using array_len_t = int64_t;

struct interchange_format {
  void write_floats_to_file(const std::string &fname) const;
  void write_bf16s_to_file(const std::string &fname) const;

  static interchange_format from_bf16_file(const std::string& fname, std::vector<array_len_t> dims);
  static interchange_format from_float_file(const std::string& fname, std::vector<array_len_t> dims);

  uint16_t* get_bf16_data() const;
  float* data = nullptr;

#ifdef USE_TORCH
  interchange_format(const torch::Tensor& data);
  torch::Tensor get_tensor() const;
#endif

private:
  std::vector<array_len_t> dims;
  array_len_t len;
  interchange_format(float *q, const std::vector<array_len_t> &r, const array_len_t &s);
};

#endif // INTERCHANGE_FORMAT_H
