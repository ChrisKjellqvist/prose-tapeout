#include "auto_allocate.h"
#include "prose_lib.h"
#include <beethoven_baremetal/fpga_handle.h>
#include <beethoven_hardware.h>

using namespace beethoven;

int main() {
  fpga_handle_t handle;
  int t_id = 0;

  prose_mh_self_attention(my_prose_allocations.input[t_id],
                          my_prose_allocations.output[t_id],
                          ModelConfig::GPTNeoConfig(1, 8), t_id, 0);
}