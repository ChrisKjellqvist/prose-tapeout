#include "auto_allocate.h"
#include "prose_lib.h"
#include "prose_vec_rptr.h"
#include <beethoven/fpga_handle.h>
#include <beethoven_hardware.h>
#include <bit>

using namespace beethoven;

int main() {
  auto config = ModelConfig::GPTNeoConfig(1, 8);
  init_alloc();
  init_rptr();
  all_layers = AllLayers();
  auto &input = transformer_h_0_mlp_input;
  printf("input:\n");
  for (int i = 0; i < 10; ++i) {
    printf("%0.4f ", as_float(((uint16_t *)input.getHostAddr())[i]));
  }
  printf("\n");
  auto &wgt = transformer_h_0_mlp_cfc_weight;
  printf("wgt:\n");
  for (int i = 0; i < 10; ++i) {
    printf("%0.4f ", as_float(((uint16_t *)wgt.getHostAddr())[i]));
  }
  printf("\n");

  auto &bias = transformer_h_0_mlp_cfc_bias;
  auto output =
      handle.malloc(2 * config.batch_size * config.seq_len * config.D * 4);
  printf("in: %llx\nwgt: %llx\nbias: %llx\nout: %llx\n", input.getFpgaAddr(),
         wgt.getFpgaAddr(), bias.getFpgaAddr(), output.getFpgaAddr());
  // void prose_g_matmul(remote_ptr const &activations, remote_ptr const
  // &weights,
  //                     remote_ptr const *norms, remote_ptr const *bias,
  //                     remote_ptr const &out, int chosen_batch_size, int M,
  //                     int K, int N, bool norm_per_batch, int biasMode);
  prose_g_matmul(input, wgt, nullptr, &bias, output, config.batch_size,
                 config.seq_len, config.D, config.D * 4, false, PROSE_biasCOLS);
  // prose_m_matmul(input, wgt, output, &bias, PROSE_biasCOLS, config.batch_size,
  //                config.seq_len, config.D, 8, true, nullptr, false, false);

  printf("output:\n");
  for (int i = 0; i < 10; ++i) {
    printf("%0.2f ", as_float(((uint16_t *)output.getHostAddr())[i]));
  }
  printf("\n");
}
