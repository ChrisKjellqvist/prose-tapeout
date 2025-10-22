#include "auto_allocate.h"
#include "prose_lib.h"
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

  prose_decoder(my_prose_allocations.input[t_id],
                my_prose_allocations.output[t_id],
                ModelConfig::GPTNeoConfig(1, 8), t_id, 0);
#ifdef LOCAL
  for (int i = 0; i < 10; ++i) {
    printf("%04x ",
           ((uint16_t *)my_prose_allocations.output[t_id].getHostAddr())[i]);
  }
#endif
}
