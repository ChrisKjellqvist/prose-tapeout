//
// Created by Christopher Kjellqvist on 1/16/24.
//

#include <cstdio>
#include <cmath>

#include "util.cc"

auto prefix = "module Exp(\n"
              "    input clock,\n"
              "    input reset,\n"
              "    input [15:0] x,\n"
              "    output [15:0] y\n"
              ");\n"
              "\n"
              "wire sign = x[15];\n"
              "wire [7:0] exponent = x[14:7];\n"
              "wire [6:0] mantissa = x[6:0];\n"
              "always @(posedge clock) begin\n"
              "    if (exponent < 8'd121) begin\n"
              "        y <= 16'h3f80;\n"
              "    end else if (exponent > 8'd132) begin\n"
              "        if (sign) begin\n"
              "            y <= 0;\n"
              "        end else begin\n"
              "            y <= 16'h7f80;\n"
              "        end\n"
              "    end else begin\n"
              "        case (x)";

auto tab = "            ";

int main() {
  FILE *f = fopen("Exp.v", "w");
  fprintf(f, "%s\n", prefix);
  for (int exponent = 121; exponent <= 132; ++exponent) {
    for (int mant = 0; mant < 128; ++mant) {
      for (int sign = 0; sign < 2; ++sign) {
        uint32_t raw = (sign << 31) | (exponent << 23) | (mant << 16);
        float x_raw = reinterpret_cast<float &>(raw);
        float y_raw = exp(x_raw);
        float y_bf16 = truncate_float_to_bf16_accuracy(y_raw);
        uint32_t y_raw_bits = reinterpret_cast<uint32_t &>(y_bf16);
        fprintf(f, "%s16'h%04x: y <= 16'h%04x;\n", tab, raw >> 16, y_raw_bits >> 16);
      }
    }
  }
  fprintf(f, "        endcase\n"
             "    end\n"
             "end\n"
             "endmodule\n");
  fclose(f);
  return 0;
}
