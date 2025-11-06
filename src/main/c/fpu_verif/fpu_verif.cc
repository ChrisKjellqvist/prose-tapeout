//
// Created by Christopher Kjellqvist on 12/22/23.
//
#include <beethoven/fpga_handle.h>
#include <beethoven_allocator_declaration.h>
#include <random>
#include <chrono>

#ifdef FPU_VERIF

const int n_cores = 1; // up to 4 currently make sense for Kria
const uint64_t total_n_samples = MAX_TESTS;
const uint64_t total_n_samples_per_core = total_n_samples / n_cores;
const uint64_t repeats_per_core = total_n_samples_per_core / MAX_TESTS;

using namespace beethoven;
fpga_handle_t handle;

int core_id = 0;
pthread_mutex_t core_id_lock = PTHREAD_MUTEX_INITIALIZER;
std::random_device rd;

#define FAIL_DRAMATICALLY 1

bool compare_equality(float f, float g, int tag) {
  bool nan_err = (isnan(f) ^ isnan(g));
  bool inf_err = (isinf(f) ^ isinf(g));
  bool val_err = (f != g);

  if (nan_err || inf_err) {
#if FAIL_DRAMATICALLY == 1
    printf("Error(%d): expected %f(%08x), got %f(%08x)\n",tag, f, reinterpret_cast<uint32_t&>(f), g
            , reinterpret_cast<uint32_t&>(g));
    handle.flush();
    exit(1);
#else
    return false;
#endif
  } else if (!isnan(f) && !isinf(f)) {
    if (val_err) {
#if FAIL_DRAMATICALLY == 1
      uint32_t hex_expected = reinterpret_cast<uint32_t&>(f);
      printf("Error(%d): expected %f (%08x), got %f\n",tag, f, hex_expected, g);
      handle.flush();
      exit(1);
#else
      return false;
#endif
    }
  }
  return true;

}

void lfsr_update_16(uint16_t &state) {
  uint16_t lsb = state & 1u;
  state >>= 1u;
  if (lsb) {
    state ^= 0xB400u;
  }
}

void lfsr_update_32(uint32_t &state) {
  uint32_t lsb = state & 1u;
  state >>= 1u;
  if (lsb) {
    state ^= 0xA3000000u;
  }
}

void * test_thread(void *) {
  // acquire core id
  pthread_mutex_lock(&core_id_lock);
  int my_core = core_id++;
  std::mt19937 gen(32);
  std::uniform_int_distribution<uint64_t> dis(0, (1ull << 48) - 1);
  auto write_addr = handle.malloc(1 << 21, READWRITE);
  pthread_mutex_unlock(&core_id_lock);
  int errors = 0;

  for (uint64_t i = 0; i < repeats_per_core; ++i) {
    // generate initializations for lfsr a_init, b_init which are 16 bits
    uint16_t a_init = dis(gen) & 0xFFFFu;
    uint16_t b_init = dis(gen) & 0xFFFFu;
    uint32_t c_init = dis(gen) & 0xFFFFFFFFu;

    // send off command and wait for it to come back. (.get() is blocking)
    BF16TestCommand(my_core, 1, a_init, b_init, c_init, write_addr).get();

    // read back the result
    auto f_ar = (float *) write_addr.getHostAddr();
    for (int j = 0; j < MAX_TESTS; ++j) {
      // generate the expected result
      uint32_t a_shift = a_init << 16, b_shift = b_init << 16;
      float a = reinterpret_cast<float&>(a_shift), b = reinterpret_cast<float&>(b_shift),
              c = reinterpret_cast<float&>(c_init);

      float expected = a * b + c;
      // compare the expected result with the result from the accelerator
      if (!compare_equality(expected, f_ar[j], j)) {
        errors++;
      }
      lfsr_update_16(a_init);
      lfsr_update_16(b_init);
      lfsr_update_32(c_init);
    }
  }
  printf("Thread %d: %d errors\n", my_core, errors);
  return nullptr;
}


int main() {
  pthread_t threads[n_cores];
  for (auto & thread : threads) pthread_create(&thread, nullptr, test_thread, nullptr);
  for (auto & thread : threads) pthread_join(thread, nullptr);
  handle.flush();
}

#else

int main() {
  printf("FPU verification tests not enabled\n");
}
#endif