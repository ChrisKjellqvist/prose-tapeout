//
// Created by Christopher Kjellqvist on 1/16/24.
//

#include <cstdio>
#include <cstdlib>
#include <random>
#include <algorithm>
#include "util.cc"
#define stringify(x) #x

auto prefix = "module Normal(\n"
              "    input clock,\n"
              "    input reset,\n"
              "    input enable,\n"
              "    input [15:0] x,\n"
              "    output [15:0] y\n"
              ");\n"
              "\n"

              "localparam SIZE = 32768;\n"
              "(* rom_style = \"distributed\" *) reg [14:0] rom [SIZE-1:0];\n"

              "(* rom_style = \"distributed\" *) reg [14:0] y_r;\n"
              "reg sign_pass;\n"
              "assign y = {sign_pass, y_r};\n"

              "initial begin\n"
              "    $readmemh(\"";
auto pre2 =  "/Normal.txt\", rom);\n"
              "end\n"

              "always @(posedge clock) begin\n"
              "   if (enable)\n"
              "   begin\n"
              "     sign_pass <= x[15];\n"
              "     y_r <= rom[x[14:0]];\n"
              "   end\n"
              "end\n"
              "endmodule\n";

auto tab = "            ";

int main(int argc, char ** argv) {
  std::random_device rd;
  std::normal_distribution<float> dis(0.0, 1.0);
  // get seed from arguments, if it's there
  long seed = 0;
  if (argc > 1) {
    seed = strtol(argv[1], nullptr, 10);
  } else {
    seed = rd();
  }

  std::mt19937 gen(seed);

  // search cwd for verilog files called "lut*.v"
  // pick the next number in the sequence and generate a gaussian lut there
  FILE * ftxt = fopen("Normal.txt", "w");
  // we'll just make the top bit the sign bit and pass it straight through
  float nums[1 << 15];
  for (float & num : nums) {
    num = std::abs(dis(gen));
  }
  // sort them to reduce entropy as much as possible (possibly easier to compress) and we're going
  // to be using a large LFSR to index into the array anyway, so no need to keep it jumbled
  std::sort(nums, nums + (1 << 15));
  for (int i = 0; i < (1 << 15); i++) {
    uint32_t raw = *(uint32_t *) &nums[i];
    fprintf(ftxt, "%04x\n", raw >> 16);
  }
  fclose(ftxt);

  FILE *f = fopen("Normal.v", "w");
  fprintf(f, "%s", prefix);
  fprintf(f, "%s", argv[1]);
  fprintf(f, "%s\n", pre2);
  fclose(f);
  return 0;
}
