#ifndef PROSE_RPTR_STRUCTURED_H
#define PROSE_RPTR_STRUCTURED_H
#include "beethoven_baremetal/allocator/alloc_baremetal.h"
#include "prose_rptr.h"

struct TransformerLayer {
  beethoven::remote_ptr ln1_w, ln1_b;
  beethoven::remote_ptr kproj_w, qproj_w, vproj_w, oproj_w, oproj_b;
  beethoven::remote_ptr ln2_w, ln2_b;
  beethoven::remote_ptr mlp_fc_w, mlp_fc_b;
  beethoven::remote_ptr mlp_proj_w, mlp_proj_b;
  constexpr TransformerLayer(int layer) {
#define for_layer(idx)                                                         \
  if (layer == idx) {                                                          \
    ln1_w = transformer_h_##idx##_ln_1_weight;                                 \
    ln1_b = transformer_h_##idx##_ln_1_bias;                                   \
    kproj_w = transformer_h_##idx##_attn_kproj_weight;                         \
    vproj_w = transformer_h_##idx##_attn_vproj_weight;                         \
    qproj_w = transformer_h_##idx##_attn_qproj_weight;                         \
    oproj_w = transformer_h_##idx##_attn_outproj_weight;                       \
    oproj_b = transformer_h_##idx##_attn_outproj_bias;                         \
    ln2_w = transformer_h_##idx##_ln_2_weight;                                 \
    ln2_b = transformer_h_##idx##_ln_2_bias;                                   \
    mlp_fc_w = transformer_h_##idx##_mlp_cfc_weight;                           \
    mlp_fc_b = transformer_h_##idx##_mlp_cfc_bias;                             \
    mlp_proj_w = transformer_h_##idx##_mlp_cproj_weight;                       \
    mlp_proj_b = transformer_h_##idx##_mlp_cproj_bias;                         \
  }
    for_layer(0) for_layer(1)
  }
  constexpr ~TransformerLayer() {}
};

#endif