#ifndef PROSE_RPTR_H
#define PROSE_RPTR_H
#include <cstdint>
#include <beethoven_baremetal/allocator/alloc_baremetal.h>
constexpr beethoven::remote_ptr DEBUG(0x0L);
constexpr beethoven::remote_ptr MASK(0x1000L);
constexpr beethoven::remote_ptr transformer_h_0_ln_1_weight(0x2000L);
constexpr beethoven::remote_ptr transformer_h_0_ln_1_bias(0x3000L);
constexpr beethoven::remote_ptr transformer_h_0_attn_kproj_weight(0x4000L);
constexpr beethoven::remote_ptr transformer_h_0_attn_vproj_weight(0x124000L);
constexpr beethoven::remote_ptr transformer_h_0_attn_qproj_weight(0x244000L);
constexpr beethoven::remote_ptr transformer_h_0_attn_outproj_weight(0x364000L);
constexpr beethoven::remote_ptr transformer_h_0_attn_outproj_bias(0x484000L);
constexpr beethoven::remote_ptr transformer_h_0_ln_2_weight(0x485000L);
constexpr beethoven::remote_ptr transformer_h_0_ln_2_bias(0x486000L);
constexpr beethoven::remote_ptr transformer_h_0_mlp_cfc_weight(0x487000L);
constexpr beethoven::remote_ptr transformer_h_0_mlp_cfc_bias(0x907000L);
constexpr beethoven::remote_ptr transformer_h_0_mlp_cproj_weight(0x909000L);
constexpr beethoven::remote_ptr transformer_h_0_mlp_cproj_bias(0xd89000L);
constexpr beethoven::remote_ptr transformer_h_1_ln_1_weight(0xd8a000L);
constexpr beethoven::remote_ptr transformer_h_1_ln_1_bias(0xd8b000L);
constexpr beethoven::remote_ptr transformer_h_1_attn_kproj_weight(0xd8c000L);
constexpr beethoven::remote_ptr transformer_h_1_attn_vproj_weight(0xeac000L);
constexpr beethoven::remote_ptr transformer_h_1_attn_qproj_weight(0xfcc000L);
constexpr beethoven::remote_ptr transformer_h_1_attn_outproj_weight(0x10ec000L);
constexpr beethoven::remote_ptr transformer_h_1_attn_outproj_bias(0x120c000L);
constexpr beethoven::remote_ptr transformer_h_1_ln_2_weight(0x120d000L);
constexpr beethoven::remote_ptr transformer_h_1_ln_2_bias(0x120e000L);
constexpr beethoven::remote_ptr transformer_h_1_mlp_cfc_weight(0x120f000L);
constexpr beethoven::remote_ptr transformer_h_1_mlp_cfc_bias(0x168f000L);
constexpr beethoven::remote_ptr transformer_h_1_mlp_cproj_weight(0x1691000L);
constexpr beethoven::remote_ptr transformer_h_1_mlp_cproj_bias(0x1b11000L);
constexpr uint32_t allocator_base(0x1b12000);
#endif
