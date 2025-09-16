#ifndef PROSE_RPTR_H
#define PROSE_RPTR_H
#include <cstdint>
#include <beethoven_baremetal/allocator/alloc_baremetal.h>
constexpr beethoven::remote_ptr DEBUG(0x0L);
constexpr beethoven::remote_ptr MASK(0x1000L);
constexpr beethoven::remote_ptr transformer_h_0_aa_input(0x2000L);
constexpr beethoven::remote_ptr transformer_h_0_attn_causal_mask(0x3000L);
constexpr beethoven::remote_ptr transformer_h_1_aa_input(0x4000L);
constexpr beethoven::remote_ptr transformer_h_1_attn_causal_mask(0x5000L);
constexpr beethoven::remote_ptr transformer_h_0_ln_1_weight(0x6000L);
constexpr beethoven::remote_ptr transformer_h_0_ln_1_bias(0x7000L);
constexpr beethoven::remote_ptr transformer_h_0_attn_kproj_weight(0x8000L);
constexpr beethoven::remote_ptr transformer_h_0_attn_vproj_weight(0x128000L);
constexpr beethoven::remote_ptr transformer_h_0_attn_qproj_weight(0x248000L);
constexpr beethoven::remote_ptr transformer_h_0_attn_outproj_weight(0x368000L);
constexpr beethoven::remote_ptr transformer_h_0_attn_outproj_bias(0x488000L);
constexpr beethoven::remote_ptr transformer_h_0_ln_2_weight(0x489000L);
constexpr beethoven::remote_ptr transformer_h_0_ln_2_bias(0x48a000L);
constexpr beethoven::remote_ptr transformer_h_0_mlp_cfc_weight(0x48b000L);
constexpr beethoven::remote_ptr transformer_h_0_mlp_cfc_bias(0x90b000L);
constexpr beethoven::remote_ptr transformer_h_0_mlp_cproj_weight(0x90d000L);
constexpr beethoven::remote_ptr transformer_h_0_mlp_cproj_bias(0xd8d000L);
constexpr beethoven::remote_ptr transformer_h_1_ln_1_weight(0xd8e000L);
constexpr beethoven::remote_ptr transformer_h_1_ln_1_bias(0xd8f000L);
constexpr beethoven::remote_ptr transformer_h_1_attn_kproj_weight(0xd90000L);
constexpr beethoven::remote_ptr transformer_h_1_attn_vproj_weight(0xeb0000L);
constexpr beethoven::remote_ptr transformer_h_1_attn_qproj_weight(0xfd0000L);
constexpr beethoven::remote_ptr transformer_h_1_attn_outproj_weight(0x10f0000L);
constexpr beethoven::remote_ptr transformer_h_1_attn_outproj_bias(0x1210000L);
constexpr beethoven::remote_ptr transformer_h_1_ln_2_weight(0x1211000L);
constexpr beethoven::remote_ptr transformer_h_1_ln_2_bias(0x1212000L);
constexpr beethoven::remote_ptr transformer_h_1_mlp_cfc_weight(0x1213000L);
constexpr beethoven::remote_ptr transformer_h_1_mlp_cfc_bias(0x1693000L);
constexpr beethoven::remote_ptr transformer_h_1_mlp_cproj_weight(0x1695000L);
constexpr beethoven::remote_ptr transformer_h_1_mlp_cproj_bias(0x1b15000L);
constexpr uint32_t allocator_base(0x1b16000);
#endif
