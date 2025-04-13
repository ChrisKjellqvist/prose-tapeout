//
// Created by Christopher Kjellqvist on 1/16/24.
//

#include <cstdio>
#include <cmath>

#include "util.cc"

auto prefix = "module GeLU(\n"
              "    input clock,\n"
              "    input reset,\n"
              "    input enable,\n"
              "    input [15:0] x,\n"
              "    output [15:0] y\n"
              ");\n"
              "\n"
              "(* rom_style = \"distributed\" *) reg [15:0] y_r;\n"
              "assign y = y_r;\n"
              "always @(posedge clock) begin\n"
              "   if (enable)\n"
              "   begin\n"
              "       case (x)";

auto tab = "          ";

int main() {
  FILE *f = fopen("GeLU.v", "w");
  fprintf(f, "%s\n", prefix);
  for (int exp = 0; exp <= 255; ++exp) {
    for (int mant = 0; mant < 128; ++mant) {
      for (int sign = 0; sign < 2; ++sign) {
        uint32_t raw = (sign << 31) | (exp << 23) | (mant << 16);
        float x_raw = reinterpret_cast<float &>(raw);
        float y_raw = gelu(x_raw);
        float y_bf16 = truncate_float_to_bf16_accuracy(y_raw);
        uint32_t y_raw_bits = reinterpret_cast<uint32_t &>(y_bf16);
        fprintf(f, "%s16'h%04x: y_r <= 16'h%04x;\n", tab, raw >> 16, y_raw_bits >> 16);
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
