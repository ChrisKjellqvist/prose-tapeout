#include <cstdint>

float truncate_float_to_bf16_accuracy(float q) {
  uint32_t q_t = reinterpret_cast<uint32_t &>(q);
  uint32_t masked = q_t & 0xFFFF0000;
  if (q_t & 0x8000) {
    masked += 0x10000;
  }
  return reinterpret_cast<float &>(masked);
}

// https://pytorch.org/docs/stable/generated/torch.nn.GELU.html
float gelu(float x) {
  return float((x * 0.5) * (1 + std::erf(x / M_SQRT2)));
}
