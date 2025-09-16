//
// Created by Chris Kjellqvist on 1/4/25.
//

#include "interchange_format.h"
#include <cstring>
#include <iostream>
#include <bit>

void interchange_format::write_floats_to_file(const std::string& fname) const {
  FILE* f = fopen(fname.c_str(), "w");
  if (!f) {
    throw std::runtime_error("Could not open file " + fname);
  }
  fwrite(this->data, sizeof(float), len, f);
  fclose(f);
  std::cout << "File written to " << fname << " with dims :";
  for (int i = 0; i < dims.size(); i++) {
    std::cout << dims[i] << " ";
  }
  std::cout << std::endl;
}

uint16_t* interchange_format::get_bf16_data() const {
  auto* buffer = new uint16_t[len];
  for (int i = 0; i < len; i++) {
    const uint32_t temp = reinterpret_cast<uint32_t&>(data[i]);
    buffer[i] = (temp >> 16) & 0xffff;
  }
  return buffer;
}


void interchange_format::write_bf16s_to_file(const std::string& fname) const {
  FILE* f = fopen(fname.c_str(), "w");
  if (!f) {
    throw std::runtime_error("Could not open file " + fname);
  }
  auto int_buff = get_bf16_data();
  fwrite(int_buff, sizeof(uint16_t), len, f);
  delete[] int_buff;
  fclose(f);
}

interchange_format interchange_format::from_bf16_file(const std::string& fname, std::vector<array_len_t> dims) {
  FILE* f = fopen(fname.c_str(), "r");
  if (!f) {
    throw std::runtime_error("Could not open file " + fname);
  }
  array_len_t len = 1;
  for (size_t i = 0; i < dims.size(); i++)
    len *= dims[i];
  auto* int_buffer = new uint16_t[len];
  fread(int_buffer, sizeof(uint16_t), len, f);
  fclose(f);

  float* data = new float[len];
  for (size_t i = 0; i < len; i++) {
    uint32_t temp = uint32_t(int_buffer[i]) << 16;
    data[i] = reinterpret_cast<float&>(temp);
  }
  delete[] int_buffer;
  return {data, dims, len};
}

interchange_format interchange_format::from_float_file(const std::string& fname, std::vector<array_len_t> dims) {
  FILE* f = fopen(fname.c_str(), "r");
  if (!f) {
    throw std::runtime_error("Could not open file " + fname);
  }
  array_len_t len = 1;
  for (size_t i = 0; i < dims.size(); i++)
    len *= dims[i];
  auto* data = new float[len];
  fread(data, sizeof(float), len, f);
  fclose(f);
  return {data, dims, len};
}

interchange_format::interchange_format(float* q, const std::vector<array_len_t> &r, const array_len_t &s) : data(q), dims(r), len(s) {}


#ifdef USE_TORCH
interchange_format::interchange_format(const torch::Tensor& data) {
  auto qdim = data.sizes();
  for (int i = 0; i < qdim.size(); i++) dims.push_back(qdim[i]);
  const auto float_ptr = data.contiguous().data_ptr<float>();
  len = 1;
  for (size_t i = 0; i < dims.size(); i++)
    len *= dims[i];
  this->data = new float[len];
  memcpy(this->data, float_ptr, len * sizeof(float));
}

torch::Tensor interchange_format::get_tensor() const {
  return torch::from_blob(this->data, at::makeArrayRef(dims));
}
#endif
