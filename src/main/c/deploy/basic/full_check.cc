#include "prose_lib.h"
#include "prose_rptr.h"
#include "prose_vec_rptr.h"
#ifdef LOCAL
#include <beethoven/fpga_handle.h>
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
  auto temp = my_prose_allocations.selfatten_intermediates[0][0];
  prose_m_matmul(input,
                 all_layers.layers[0].proj_wgts[0].kproj,
                 temp, nullptr, 0,
                 1, 32, 768, 768, 1, nullptr, 0, 0);
#ifdef LOCAL
  printf("output:\n");
  for (int i = 0; i < 32; ++i) {
    printf("%04x ", ((uint16_t*)(temp.getHostAddr()))[i]);
  }
  printf("\ninput:\n");
  for (int i = 0; i < 32; ++i) {
    printf("%04x ", ((uint16_t*)(input.getHostAddr()))[i]);
  }
  printf("\nweights:\n");
  for (int i = 0; i < 32; ++i) {
    printf("%04x ", ((uint16_t*)(all_layers.layers[0].proj_wgts[0].kproj.getHostAddr()))[i]);
  }
  printf("\n");
#endif
}
