#ifndef AUTO_ALLOCATE_H
#define AUTO_ALLOCATE_H

#include <beethoven_baremetal/allocator/alloc_baremetal.h>
#include <cassert>
#include <prose_rptr.h>
#ifdef LOCAL
extern beethoven::fpga_handle_t handle;
#endif

template <uint32_t n_threads, uint32_t dim, uint32_t batch_size,
          uint32_t context_length, uint32_t n_heads>
struct prose_allocations {
  beethoven::remote_ptr input[n_threads];
  beethoven::remote_ptr ln_out[n_threads];
  beethoven::remote_ptr selfatten_intermediates[n_threads][4];
  beethoven::remote_ptr selfatten_attenscore[n_threads];
  beethoven::remote_ptr mlp_intermediate[n_threads];
  beethoven::remote_ptr output[n_threads];
  constexpr prose_allocations() {};
  constexpr prose_allocations(
      const beethoven::remote_ptr (&input)[n_threads],
      const beethoven::remote_ptr (&output)[n_threads],
      const beethoven::remote_ptr (&selfatten_intermediates)[n_threads][4],
      const beethoven::remote_ptr (&ln_out)[n_threads],
      const beethoven::remote_ptr (&mlp_intermediate)[n_threads],
      const beethoven::remote_ptr (&attenmatrix)[n_threads]) {
    for (int i = 0; i < n_threads; ++i) {
      this->output[i] = output[i];
      this->input[i] = input[i];
      for (int j = 0; j < 4; ++j) {
        this->selfatten_intermediates[i][j] = selfatten_intermediates[i][j];
      }
      this->ln_out[i] = ln_out[i];
      this->mlp_intermediate[i] = mlp_intermediate[i];
      this->selfatten_attenscore[i] = attenmatrix[i];
    }
  }
  __constructor_annot__ ~prose_allocations() {};
};

namespace auto_alloc {

constexpr inline uint32_t alloc(uint32_t &allocator, uint32_t sz) {
  uint32_t ret = allocator;
  allocator += sz;
  return ret;
}

constexpr inline void align_to_4K(uint32_t &allocator) {
  if (allocator % (4 * 1024) != 0) {
    allocator += ((4 * 1024) - (allocator % (4 * 1024)));
  }
}

template <uint32_t n_threads, uint32_t dim, uint32_t batch_size,
          uint32_t context_length, uint32_t n_heads>
#ifndef KRIA
constexpr
#endif
    prose_allocations<n_threads, dim, batch_size, context_length, n_heads>
    get_prose_allocs() {
  uint32_t head_size = dim / n_heads;
  auto allocator = allocator_base;
  beethoven::remote_ptr input[n_threads];
  beethoven::remote_ptr output[n_threads];
  beethoven::remote_ptr ln_out[n_threads];
  beethoven::remote_ptr selfatten_intermediates[n_threads][4];
  beethoven::remote_ptr mlp_intermediate[n_threads];
  beethoven::remote_ptr attenscore[n_threads];
  #ifndef KRIA
#define ALLOC(amt) (beethoven::remote_ptr(alloc(allocator, (amt))))
#else
#define ALLOC(amt) (handle.malloc(amt))
#endif

  for (auto i = 0; i < n_threads; ++i) {
    input[i] = ALLOC(batch_size * dim * context_length * 2);
    output[i] = ALLOC(batch_size * dim * context_length * 2);
    ln_out[i] = ALLOC(batch_size * dim * context_length * 2);
    mlp_intermediate[i] = ALLOC(batch_size * dim * 4 * context_length * 2);
    for (auto j = 0; j < 3; ++j) {
      selfatten_intermediates[i][j] =
          ALLOC(batch_size * head_size * context_length * 2);
    }
    selfatten_intermediates[i][3] = ALLOC(2 * dim);
    attenscore[i] = ALLOC(2 * batch_size * context_length * context_length);
  }

  assert(allocator < 32 * 1024 * 1024);
  return prose_allocations<n_threads, dim, batch_size, context_length, n_heads>(
      input, output, selfatten_intermediates, ln_out, mlp_intermediate, attenscore);
}
} // namespace auto_alloc

extern const prose_allocations<1, 768, 1, 16, 12> my_prose_allocations;

#endif