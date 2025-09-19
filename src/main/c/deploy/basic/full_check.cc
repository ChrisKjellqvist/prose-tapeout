#include "prose_lib.h"
#include "prose_rptr.h"
#include "prose_vec_rptr.h"
#include <beethoven_baremetal/fpga_handle.h>
#include <beethoven_hardware.h>

#include "auto_allocate.h"

using namespace beethoven;

int main() {
  fpga_handle_t handle;
  prose_m_matmul(my_prose_allocations.input[0],
                 all_layers.layers[0].proj_wgts[0].kproj,
                 my_prose_allocations.selfatten_intermediates[0][0], nullptr, 0,
                 1, 32, 768, 768, 1, nullptr, 0, 0);
}