#include "auto_allocate.h"
#include "prose_lib.h"
#include "prose_lib_twt.h"
#ifdef LOCAL
#include <beethoven/fpga_handle.h>
#else
#include <beethoven_baremetal/fpga_handle.h>
#endif
#include "prose_vec_rptr.h"
#include <beethoven_hardware.h>

using namespace beethoven;

int main() {
  int t_id = 0;

#ifdef LOCAL
  init_alloc();
  init_rptr();
  all_layers = AllLayers();
#endif
  auto &input = transformer_h_0_aa_input;
  decoder_scheduler test(input, ModelConfig::GPTNeoConfig(1, 8), 
                my_prose_allocations.output[t_id], t_id, 0);
  test.execute();
}
