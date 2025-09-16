#include <beethoven_baremetal/fpga_handle.h>
#include <beethoven_hardware.h>
#include "prose_lib.h"
#include "auto_allocate.h"

using namespace beethoven;

extern const prose_allocations<1, 768, 1, 16, 12> my_prose_allocations;

int main() {
  fpga_handle_t handle;
  int t_id = 0;
  prose_self_attention(my_prose_allocations.input[t_id], 1, 16, 768, 64, my_prose_allocations.input[t_id], t_id, 0);
}