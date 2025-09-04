#include <cstdio>
#include <cstdlib>
#include <random>
#include <algorithm>
#include "util.cc"

std::string prefix =
         "module Normal (\n"
         "    input clock,\n"
         "    input reset,\n"
         "    input enable,\n"
         "    input [15:0] x,\n"
         "    output [15:0] y\n"
         ");\n"
         "\n"
         "(* rom_style = \"distributed\" *) reg [14:0] y_reg;\n"
         "reg sign_pass;\n"
         "assign y = {sign_pass, y_reg};\n"
         "always @(posedge clock) begin\n"
         "    sign_pass <= x[15];\n"
         "    case (x[14:0])\n";

std::string tab() {
  return "      ";
}

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
  FILE * f = fopen("Normal.v", "w");
  fprintf(f, "%s", prefix.c_str());
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
    fprintf(f, "%s15'h%04x: y_reg <= 16'h%04x;\n", tab().c_str(), i, raw >> 16);
  }
  fprintf(f, "    endcase\n"
             "end\n"
             "endmodule\n");
  fclose(f);


}
