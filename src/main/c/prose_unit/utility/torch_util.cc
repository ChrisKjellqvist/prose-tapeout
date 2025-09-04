//
// Created by Chris Kjellqvist on 1/4/25.
//

#ifdef USE_TORCH
#include "torch_util.h"
#include <fstream>
#include <iostream>

torch::Tensor load_tensor(const std::string& filename) {
  std::ifstream file(filename, std::ios::binary);
  std::vector<char> data((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
  auto ivalue = torch::pickle_load(data);
  // get the tensor from the ivalue
  auto tensor = ivalue.toTensor();
  std::cout << "Loaded tensor" << filename << " with shape: " << tensor.sizes() << std::endl << std::flush;
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
  int rows_to_print = std::min(10, (int)tensor.size(0));
  int cols_to_print = std::min(10, (int)tensor.size(1));

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
  int rows_to_print = std::min(10, (int)tensor.size(0));

  std::cout << "First 10x10 elements of the output tensor:" << std::endl;
  for (int i = 0; i < rows_to_print; ++i) {
    std::cout << tensor[i].item<float>() << " ";
  }
  std::cout << std::endl;
}

void print_2d_tensor(torch::Tensor tensor) {
  // Print the first 10x10 elements of the output tensor
  int rows_to_print = (int)tensor.size(0);
  int cols_to_print = (int)tensor.size(1);
  for (int i = 0; i < rows_to_print; ++i) {
    for (int j = 0; j < cols_to_print; ++j) {
      std::cout << tensor[i][j].item<float>() << " ";
    }
    std::cout << std::endl;
  }
}

void print_batched_2d_tensor(const torch::Tensor& tensor) {
  // Print the first 10x10 elements of the output tensor
  int batch_size = (int)tensor.size(0);
  int rows_to_print = (int)tensor.size(1);
  int cols_to_print = (int)tensor.size(2);
  for (int b = 0; b < batch_size; ++b) {
    std::cout << "Batch " << b << std::endl;
    for (int i = 0; i < rows_to_print; ++i) {
      for (int j = 0; j < cols_to_print; ++j) {
        float item = tensor[b][i][j].item<float>();
        uint32_t cast = reinterpret_cast<uint32_t&>(item);
        uint16_t bf16 = (uint16_t)(cast >> 16);
        printf("%0.2f(%04x)\t", item, bf16);
      }
      std::cout << std::endl;
    }
  }
}

void print_1d_tensor(torch::Tensor tensor) {
  // Print the first 10x10 elements of the output tensor
  int cols_to_print = (int)tensor.size(0);
  for (int i = 0; i < cols_to_print; ++i) {
    std::cout << tensor[i].item<float>() << " ";
  }
  std::cout << std::endl;
}

void print_1d_tensor_as_16hex(torch::Tensor tensor) {
  // Print the first 10x10 elements of the output tensor
  int cols_to_print = (int)tensor.size(0);
  for (int i = 0; i < cols_to_print; ++i) {
    float x = tensor[i].item<float>();
    uint32_t cast = reinterpret_cast<uint32_t&>(x);
    printf("%04x ", (uint16_t)(cast >> 16));
  }
  std::cout << std::endl;
}

float max_pct_diff(const torch::Tensor& a, const torch::Tensor& b) {
  auto diff = (a - b).abs();
  auto max_diff = diff.max().item<float>();
  auto max_a = a.abs().max().item<float>();
  auto max_b = b.abs().max().item<float>();
  return max_diff / std::max(max_a, max_b) * 100;
}

torch::Tensor* multi_head_torch_tensor_to_flt_array(const torch::Tensor& tensor, int n_heads, int slice_dim) {
  auto* array = new torch::Tensor[n_heads];
  auto dims = tensor.sizes();
  size_t dim_of_targetDim = dims[slice_dim];
  if (dim_of_targetDim % n_heads != 0)
    throw std::runtime_error("slicing is fucked up");
  size_t adjusted_dim = dim_of_targetDim / n_heads;
  size_t other_dims = 1;
  for (int i = 0; i < dims.size(); ++i) {
    if (i == slice_dim)
      continue;
    other_dims *= dims[i];
  }

  for (int h = 0; h < n_heads; ++h) {
    auto slice = tensor.slice(slice_dim, h * adjusted_dim, (h + 1) * adjusted_dim).contiguous().clone();
    array[h] = slice;
    std::cout << "dim before: ";
    for (int i = 0; i < dims.size(); ++i) {
      std::cout << dims[i] << " ";
    }
    std::cout << "\t\tdim after: ";
    for (int i = 0; i < dims.size(); ++i) {
      std::cout << slice.sizes()[i] << " ";
    }
    std::cout << std::endl;
  }

  return array;
}

#endif
