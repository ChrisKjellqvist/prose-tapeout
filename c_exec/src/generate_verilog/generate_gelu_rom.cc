//
// Created by Christopher Kjellqvist on 1/16/24.
//

#include <cstdio>
#include <cmath>

#include "util.cc"
#define stringify(x) #x

auto prefix = "module GeLU(\n"
              "    input clock,\n"
              "    input reset,\n"
              "    input enable,\n"
              "    input [15:0] x,\n"
              "    output [15:0] y\n"
              ");\n"
              "\n"

              "localparam SIZE = 65536;\n"
              "(* rom_style = \"distributed\" *) reg [15:0] rom [SIZE-1:0];\n"

              "(* rom_style = \"distributed\" *) reg [15:0] y_r;\n"
              "assign y = y_r;\n"

              "initial begin\n"
              "    $readmemh(\"";
auto pre2 = "/GeLU.txt\", rom);\n"
              "end\n"

              "always @(posedge clock) begin\n"
              "   if (enable)\n"
              "   begin\n"
              "       y_r <= rom[x];\n"
              "   end\n"
              "end\n"
              "endmodule\n";

auto tab = "            ";

int main(int argc, char ** argv) {
  FILE *ftxt = fopen("GeLU.txt", "w");
        for (int sign = 0; sign < 2; ++sign) {

  for (int exp = 0; exp <= 255; ++exp) {
    for (int mant = 0; mant < 128; ++mant) {
          uint32_t raw = (sign << 31) | (exp << 23) | (mant << 16);
          float x_raw = reinterpret_cast<float &>(raw);
          float y_raw = gelu(x_raw);
          float y_bf16 = truncate_float_to_bf16_accuracy(y_raw);
          uint32_t y_raw_bits = reinterpret_cast<uint32_t &>(y_bf16);
          fprintf(ftxt, "%04x\n", y_raw_bits >> 16);
        }
      }
   }
  fclose(ftxt);

  FILE *f = fopen("GeLU.v", "w");
  fprintf(f, "%s", prefix);
  fprintf(f, "%s", argv[1]);
  fprintf(f, "%s\n", pre2);
  fclose(f);
  return 0;
}
