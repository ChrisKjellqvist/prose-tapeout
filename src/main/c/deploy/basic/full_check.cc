#include "prose_lib.h"
#include "prose_rptr.h"
#include "prose_vec_rptr.h"
#ifdef LOCAL
#include <beethoven/fpga_handle.h>
#include <bit>
#else
#include <beethoven_baremetal/fpga_handle.h>
#endif
#include <beethoven_hardware.h>

#include "auto_allocate.h"

using namespace beethoven;

int main() {
#ifdef LOCAL
  init_alloc();
  init_rptr();
  all_layers = AllLayers();
#endif
  auto input = transformer_h_0_aa_input;
  auto wgts = all_layers.layers[0].proj_wgts[0].kproj;
  auto temp = my_prose_allocations.selfatten_intermediates[0][0];
#ifdef LOCAL
  printf("input: %llx\n", input.getFpgaAddr());
  printf("wgt: %llx\n", wgts.getFpgaAddr()); 
  printf("out: %llx\n", temp.getFpgaAddr());
#endif
  prose_m_matmul(input,
                 wgts,
                 temp, nullptr, 0,
                 1, 4, 768, 64, 1, nullptr, 0, 0);
#ifdef LOCAL
  printf("output:\n");
  for (int i = 0; i < 32; ++i) {
    printf("%0.2f ", as_float(((uint16_t*)(temp.getHostAddr()))[i]));
  }
  printf("\ninput:\n");
  for (int i = 0; i < 32; ++i) {
    printf("%0.2f ", as_float(((uint16_t*)(input.getHostAddr()))[i]));
  }
  printf("\nweights:\n");
  for (int i = 0; i < 32; ++i) {
    printf("%0.2f ", as_float(((uint16_t*)(all_layers.layers[0].proj_wgts[0].kproj.getHostAddr()))[i]));
  }
  printf("\n");
#endif
}
