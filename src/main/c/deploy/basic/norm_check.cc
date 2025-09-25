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

// assume LOCAL for now
int main() {
  int v_len = 32;
  int batch_size = 1;

  auto wb = handle.malloc(v_len * 2 * 2);
  auto input = handle.malloc(v_len * 2);
  auto output = handle.malloc(v_len * 2);

  Norm::norm(0, wb, input, 1, batch_size, 1.0 / v_len, flagLayerNorm, output, 1,
             v_len)
      .get();
}
