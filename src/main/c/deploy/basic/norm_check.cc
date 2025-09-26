#include "auto_allocate.h"
#include "prose_lib.h"
#include <beethoven/fpga_handle.h>
#include "prose_vec_rptr.h"
#include <beethoven_hardware.h>

using namespace beethoven;

uint16_t asbf16(float f) {
  return uint16_t(std::bit_cast<uint32_t>(f) >> 16);
}

float asfloat(uint16_t i) {
  return std::bit_cast<float>(uint32_t(i) << 16);
}

float golden_mean(uint16_t *p, int l) {
  float sum = 0;
  for (int i = 0; i < l; ++i) {
    sum += asfloat(p[i]);
  }
  return sum / l;
}

float golden_var(uint16_t *p, int l, float m) {
  float sum = 0;

  for (int i = 0; i < l; ++i) {
    auto diff = asfloat(p[i]) - m;
    sum += diff * diff;
  }

  return sum / l;
}

float *golden_ln(uint16_t *p, int l, uint16_t *gamma_beta) {
  float mean = golden_mean(p, l);
  float var = golden_var(p, l, mean);
  float *out = new float[l];
  float epsilon = 1e-5;
  for (int i = 0; i < l; ++i) {
    out[i] = (asfloat(p[i]) - mean) / std::sqrtf(var + epsilon) * asfloat(gamma_beta[2*i]) + asfloat(gamma_beta[2*i+1]); 
  }
  return out;
}

// assume LOCAL for now
int main() {
  int v_len = 32;
  int batch_size = 1;

  auto gam_bet = handle.malloc(v_len * 2 * 2);
  auto input = handle.malloc(v_len * 2);
  auto output = handle.malloc(v_len * 2);

  auto gb_ptr = (uint16_t*) gam_bet.getHostAddr();
  auto i_ptr = (uint16_t*) input.getHostAddr();
  auto o_ptr = (uint16_t*) output.getHostAddr();

  for (int i = 0; i < v_len; ++i) {
    gb_ptr[i*2] = asbf16(float(i));
    gb_ptr[i*2+1] = asbf16(float(i+1));
    i_ptr[i] = asbf16(i / 2);
    o_ptr[i] = 0;
  }

  Norm::norm(0, gam_bet, input, 1, batch_size, 1.0 / v_len, flagLayerNorm, output, 1,
             v_len)
      .get();

  auto golden = golden_ln(i_ptr, v_len, gb_ptr);
  for (int i = 0; i < v_len; ++i) {
    printf("GOLD(%0.2f) HW(%0.2f)\n", golden[i], asfloat(o_ptr[i]));
  }
}
