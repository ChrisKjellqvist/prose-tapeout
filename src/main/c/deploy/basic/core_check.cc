#include <beethoven_baremetal/fpga_handle.h>
#include <beethoven_hardware.h>
#include "prose_rptr.h"

#include "auto_allocate.h"

using namespace beethoven;

constinit const prose_allocations<1, 768, 1, 16, 12> my_prose_allocations = auto_alloc::get_prose_allocs<1, 768, 1, 16, 12>();

int main() {
  fpga_handle_t handle;
}