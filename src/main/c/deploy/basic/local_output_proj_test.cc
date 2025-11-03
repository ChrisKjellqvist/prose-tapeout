#include "prose_lib.h"
#include "prose_vec_rptr.h"
#include "auto_allocate.h"
#include <beethoven/fpga_handle.h>
#include <bit>
#include <beethoven_hardware.h>

using namespace beethoven;

int main() {
  auto config = ModelConfig::GPTNeoConfig(1, 8);
  init_alloc();
  init_rptr();
  all_layers = AllLayers();
  auto &input = transformer_h_0_attn_output_test_input;
  auto &wgt = transformer_h_0_attn_outproj_weight;
  auto &bias = transformer_h_0_attn_outproj_bias;
  auto output = handle.malloc(2 * config.batch_size * config.seq_len * config.D);
  printf("in: %llx\nwgt: %llx\nbias: %llx\nout: %llx\n", input.getFpgaAddr(),
   wgt.getFpgaAddr(), bias.getFpgaAddr(), output.getFpgaAddr());
  prose_m_matmul(input, wgt, output, &bias, PROSE_biasCOLS,
                 config.batch_size, config.seq_len, config.D, config.D,
                 true, nullptr, false, false);
  for (int i = 0; i < 10; ++i) {
    printf("%0.2f ", as_float(((uint16_t*)output.getHostAddr())[i]));
  }
  
}
