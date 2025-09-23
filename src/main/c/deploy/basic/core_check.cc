#include "auto_allocate.h"
#include "prose_lib.h"
#ifdef LOCAL
#include <beethoven/fpga_handle.h>
#else
#include <beethoven_baremetal/fpga_handle.h>
#endif
#include <beethoven_hardware.h>

using namespace beethoven;

int main() {
  int t_id = 0;

  prose_mh_self_attention(my_prose_allocations.input[t_id],
                          my_prose_allocations.output[t_id],
                          ModelConfig::GPTNeoConfig(1, 8), t_id, 0);
}
