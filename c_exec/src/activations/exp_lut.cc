#include "torch/torch.h"
#include <iostream>
#include <cstring>

uint16_t torch_bfloat16_exp(uint16_t input_raw_bits) {
    // Create a tensor from the float
    auto input_tensor = torch::zeros({1}, torch::dtype(torch::kBFloat16));

    // memcpy
    std::memcpy(input_tensor.data_ptr(), &input_raw_bits, sizeof(input_raw_bits));

    // Apply GeLU
    auto output_tensor = torch::exp(input_tensor);

    // memcpy
    uint16_t output_raw_bits;
    std::memcpy(&output_raw_bits, output_tensor.data_ptr(), sizeof(output_raw_bits));

    return output_raw_bits;
}


int main() {
    // iterate through all possible uint16_t values
    long largest_pos_diff = 0;
    long smallest_neg_diff = 0;
    for (uint16_t i = 0; i < 0xffff; i++) {
        uint16_t output_raw_bits = torch_bfloat16_exp(i);
        std::string out_str;
//        if (output_raw_bits == 32768 || output_raw_bits == 32704) {
//            out_str = "inf";
//        } else if (output_raw_bits == 0) {
//            out_str = "zero";
//        } else {
//            out_str = std::to_string(output_raw_bits);
//        }
        out_str = std::to_string(output_raw_bits);
        std::cout << i << "," << out_str << std::endl;
    }

    return 0;
}
