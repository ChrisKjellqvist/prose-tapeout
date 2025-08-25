#include <beethoven/fpga_handle.h>
#include <beethoven_allocator_declaration.h>
#include <cstring>
#include "torch/torch.h"
#include <iostream>
#include <cstring>
using namespace beethoven;

uint16_t torch_bfloat16_gelu(uint16_t input_raw_bits) {
    // Create a tensor from the float
    auto input_tensor = torch::zeros({1}, torch::dtype(torch::kBFloat16));

    // memcpy
    std::memcpy(input_tensor.data_ptr(), &input_raw_bits, sizeof(input_raw_bits));

    // Apply GeLU
    auto output_tensor = torch::nn::functional::gelu(input_tensor);

    // memcpy
    uint16_t output_raw_bits;
    std::memcpy(&output_raw_bits, output_tensor.data_ptr(), sizeof(output_raw_bits));

    return output_raw_bits;
}

int main() {
    fpga_handle_t fpga;
    int all_passed = 1;
    for (uint16_t i = 0; i < 65535; i++) {
        uint16_t torch_result = torch_bfloat16_gelu(i);
        uint16_t fpga_result;
        auto fpga_raw = MyGELUCommand(0, i).get().out;
        std::memcpy(&fpga_result, &fpga_raw, sizeof(fpga_result));
        if (torch_result != fpga_result) {
            printf("Mismatch at %hx: torch %hx, fpga %hx\n", i, torch_result, fpga_result);
            all_passed = 0;
        }
        if (i % 1000 == 0) {
            printf("%d\n", i);
        }
    }
    if (all_passed) {
        printf("All passed!\n");
    }
}
