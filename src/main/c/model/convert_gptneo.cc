//
// Created by Christopher Kjellqvist on 4/4/24.
//

#include "../utility/util.h"
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

float *load_matrix(const std::string &filename, int rows, int cols) {
  std::ifstream file(filename, std::ios::binary);
  file.seekg(128, std::ios::beg);
  float *matrix = new float[rows * cols];
  file.read(reinterpret_cast<char *>(matrix), rows * cols * sizeof(float));
  file.close();
  return matrix;
}

std::vector<std::pair<std::string, std::vector<int>>> get_index() {
  // Load the index file
  std::ifstream file("gpt_neo/layer_dict.txt");
  std::vector<std::pair<std::string, std::vector<int>>> index;
  // for each line in file, it is a file name and then space separated integers
  std::string line;
  index.push_back({"DEBUG", {1024}});
  while (std::getline(file, line)) {
    std::istringstream iss(line);
    std::string name;
    iss >> name;
    std::vector<int> shape;
    int dim;
    while (iss >> dim) {
      shape.push_back(dim);
    }
    index.emplace_back(name, shape);
  }
  return index;
}

uint16_t *convertFloatToBF16Vector(const float *input, int size) {
  uint16_t *output = new uint16_t[size];
  for (int i = 0; i < size; ++i) {
    uint32_t act_hex = reinterpret_cast<const uint32_t &>(input[i]);
    act_hex &= 0xffff0000;
    output[i] = uint16_t(act_hex >> 16);
  }
  return output;
}

const int tile_size = 4;

int main() {
  auto index = get_index();
  for (const auto &pair : index) {
    std::cout << pair.first << " ";
    for (int dim : pair.second) {
      std::cout << dim << " ";
    }
    std::cout << std::endl;
  }

  std::vector<std::pair<uint16_t *, int>> matrices;

  for (const auto &pair : index) {
    int cols = pair.second[0];
    int rows;
    if (pair.second.size() != 2) {
      rows = 1;
    } else {
      rows = pair.second[1];
    }
    if (pair.first == "DEBUG") {
      int n_items = 1024 / sizeof(uint32_t);
      uint32_t *debug = new uint32_t[n_items];
      for (int i = 0; i < n_items; ++i) {
        switch (i % 4) {
        case 0:
          debug[i] = 0xDEADBEEF;
          break;
        case 1:
          debug[i] = 0xABCD1234;
          break;
        case 2:
          debug[i] = 0xFFEEEEDD;
          break;
        case 3:
          debug[i] = 0x01020304;
          break;
        default:
          throw std::runtime_error("?");
        }
      }
      matrices.emplace_back((uint16_t*)debug, 1024 / 2);
    } else {
      auto matrix = load_matrix("gpt_neo/" + pair.first + ".npy", rows, cols);
      auto bf16_matrix = convertFloatToBF16Vector(matrix, rows * cols);
      auto prose_form = new uint16_t[rows * cols];
      convertRowMajorFormatToProSERowMajor(bf16_matrix, prose_form, rows, cols,
                                           1, tile_size);
      matrices.emplace_back(prose_form, rows * cols);
    }
  }

  std::vector<std::string> names_only;
  names_only.reserve(index.size());
  for (const auto &pair : index) {
    names_only.push_back(pair.first);
  }

  write_to_file("gpt_neo/prose_input.bin", matrices,
                {std::make_pair("gpt_neo/prose_index.txt", names_only)});

  // free matrices
  for (const auto &pair : matrices) {
    delete[] pair.first;
  }
}
