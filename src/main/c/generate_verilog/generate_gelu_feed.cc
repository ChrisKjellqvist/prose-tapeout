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
              "wire [7:0] exp = x[14:7];\n"
              "wire [14:0] important_bits = x[14:0];"
              "wire sign = x[15];"
              "assign y = y_r;\n"
              "always @(posedge clock) begin\n"
              "   if (enable)\n"
              "   begin\n"
              "       if (exp >= 130)\n"
              "       begin\n"
              "           y_r <= 0;\n"
              "       end else begin\n"
              "           case (important_bits)\n";
auto tab = "              ";

int main() {
  FILE *f = fopen("GeLU.v", "w");
  fprintf(f, "%s\n", prefix);
  for (int exp = 0; exp < 130; ++exp) {
    for (int mant = 0; mant < 128; ++mant) {
        uint32_t raw = (exp << 23) | (mant << 16);
        float x_raw = reinterpret_cast<float &>(raw);
        float relu = (x_raw > 0)? x_raw : 0;
        float y_raw = gelu(x_raw) - relu;
        float y_bf16 = truncate_float_to_bf16_accuracy(y_raw);
        uint32_t y_raw_bits = reinterpret_cast<uint32_t &>(y_bf16);
        fprintf(f, "%s16'h%04x: y_r <= 16'h%04x;\n", tab, raw >> 16, y_raw_bits >> 16);
    }
  }
  fprintf(f, "            endcase\n"
             "        end\n"
             "    end\n"
             "end\n"
             "endmodule\n");
  fclose(f);
  return 0;
}
