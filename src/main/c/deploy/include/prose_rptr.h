#ifndef PROSE_RPTR_H
#define PROSE_RPTR_H
#include <cstdint>
#ifdef LOCAL
#include <beethoven/allocator/alloc.h>
#else
#include <beethoven_baremetal/allocator/alloc_baremetal.h>
#endif
#include "prose_lib.h"

void init_rptr();
__ptr_annot__ beethoven::remote_ptr DEBUG PTR_FROM_OFFSET_H(0x0L, 0x400L);
__ptr_annot__ beethoven::remote_ptr transformer_h_0_aa_input PTR_FROM_OFFSET_H(0x1000L, 0x3000L);
__ptr_annot__ beethoven::remote_ptr transformer_h_0_attn_causal_mask PTR_FROM_OFFSET_H(0x4000L, 0x80L);
__ptr_annot__ beethoven::remote_ptr transformer_h_0_attn_output_test_input PTR_FROM_OFFSET_H(0x5000L, 0x3000L);
__ptr_annot__ beethoven::remote_ptr transformer_h_0_mlp_input PTR_FROM_OFFSET_H(0x8000L, 0x3000L);
__ptr_annot__ beethoven::remote_ptr transformer_h_1_aa_input PTR_FROM_OFFSET_H(0xb000L, 0x3000L);
__ptr_annot__ beethoven::remote_ptr transformer_h_1_attn_causal_mask PTR_FROM_OFFSET_H(0xe000L, 0x80L);
__ptr_annot__ beethoven::remote_ptr transformer_h_1_attn_output_test_input PTR_FROM_OFFSET_H(0xf000L, 0x3000L);
__ptr_annot__ beethoven::remote_ptr transformer_h_1_mlp_input PTR_FROM_OFFSET_H(0x12000L, 0x3000L);
__ptr_annot__ beethoven::remote_ptr transformer_h_0_ln_1_gb PTR_FROM_OFFSET_H(0x15000L, 0xc00L);
__ptr_annot__ beethoven::remote_ptr transformer_h_0_attn_kproj_weight_h0 PTR_FROM_OFFSET_H(0x16000L, 0x18000L);
__ptr_annot__ beethoven::remote_ptr transformer_h_0_attn_kproj_weight_h1 PTR_FROM_OFFSET_H(0x2e000L, 0x18000L);
__ptr_annot__ beethoven::remote_ptr transformer_h_0_attn_kproj_weight_h2 PTR_FROM_OFFSET_H(0x46000L, 0x18000L);
__ptr_annot__ beethoven::remote_ptr transformer_h_0_attn_kproj_weight_h3 PTR_FROM_OFFSET_H(0x5e000L, 0x18000L);
__ptr_annot__ beethoven::remote_ptr transformer_h_0_attn_kproj_weight_h4 PTR_FROM_OFFSET_H(0x76000L, 0x18000L);
__ptr_annot__ beethoven::remote_ptr transformer_h_0_attn_kproj_weight_h5 PTR_FROM_OFFSET_H(0x8e000L, 0x18000L);
__ptr_annot__ beethoven::remote_ptr transformer_h_0_attn_kproj_weight_h6 PTR_FROM_OFFSET_H(0xa6000L, 0x18000L);
__ptr_annot__ beethoven::remote_ptr transformer_h_0_attn_kproj_weight_h7 PTR_FROM_OFFSET_H(0xbe000L, 0x18000L);
__ptr_annot__ beethoven::remote_ptr transformer_h_0_attn_kproj_weight_h8 PTR_FROM_OFFSET_H(0xd6000L, 0x18000L);
__ptr_annot__ beethoven::remote_ptr transformer_h_0_attn_kproj_weight_h9 PTR_FROM_OFFSET_H(0xee000L, 0x18000L);
__ptr_annot__ beethoven::remote_ptr transformer_h_0_attn_kproj_weight_h10 PTR_FROM_OFFSET_H(0x106000L, 0x18000L);
__ptr_annot__ beethoven::remote_ptr transformer_h_0_attn_kproj_weight_h11 PTR_FROM_OFFSET_H(0x11e000L, 0x18000L);
__ptr_annot__ beethoven::remote_ptr transformer_h_0_attn_vproj_weight_h0 PTR_FROM_OFFSET_H(0x136000L, 0x18000L);
__ptr_annot__ beethoven::remote_ptr transformer_h_0_attn_vproj_weight_h1 PTR_FROM_OFFSET_H(0x14e000L, 0x18000L);
__ptr_annot__ beethoven::remote_ptr transformer_h_0_attn_vproj_weight_h2 PTR_FROM_OFFSET_H(0x166000L, 0x18000L);
__ptr_annot__ beethoven::remote_ptr transformer_h_0_attn_vproj_weight_h3 PTR_FROM_OFFSET_H(0x17e000L, 0x18000L);
__ptr_annot__ beethoven::remote_ptr transformer_h_0_attn_vproj_weight_h4 PTR_FROM_OFFSET_H(0x196000L, 0x18000L);
__ptr_annot__ beethoven::remote_ptr transformer_h_0_attn_vproj_weight_h5 PTR_FROM_OFFSET_H(0x1ae000L, 0x18000L);
__ptr_annot__ beethoven::remote_ptr transformer_h_0_attn_vproj_weight_h6 PTR_FROM_OFFSET_H(0x1c6000L, 0x18000L);
__ptr_annot__ beethoven::remote_ptr transformer_h_0_attn_vproj_weight_h7 PTR_FROM_OFFSET_H(0x1de000L, 0x18000L);
__ptr_annot__ beethoven::remote_ptr transformer_h_0_attn_vproj_weight_h8 PTR_FROM_OFFSET_H(0x1f6000L, 0x18000L);
__ptr_annot__ beethoven::remote_ptr transformer_h_0_attn_vproj_weight_h9 PTR_FROM_OFFSET_H(0x20e000L, 0x18000L);
__ptr_annot__ beethoven::remote_ptr transformer_h_0_attn_vproj_weight_h10 PTR_FROM_OFFSET_H(0x226000L, 0x18000L);
__ptr_annot__ beethoven::remote_ptr transformer_h_0_attn_vproj_weight_h11 PTR_FROM_OFFSET_H(0x23e000L, 0x18000L);
__ptr_annot__ beethoven::remote_ptr transformer_h_0_attn_qproj_weight_h0 PTR_FROM_OFFSET_H(0x256000L, 0x18000L);
__ptr_annot__ beethoven::remote_ptr transformer_h_0_attn_qproj_weight_h1 PTR_FROM_OFFSET_H(0x26e000L, 0x18000L);
__ptr_annot__ beethoven::remote_ptr transformer_h_0_attn_qproj_weight_h2 PTR_FROM_OFFSET_H(0x286000L, 0x18000L);
__ptr_annot__ beethoven::remote_ptr transformer_h_0_attn_qproj_weight_h3 PTR_FROM_OFFSET_H(0x29e000L, 0x18000L);
__ptr_annot__ beethoven::remote_ptr transformer_h_0_attn_qproj_weight_h4 PTR_FROM_OFFSET_H(0x2b6000L, 0x18000L);
__ptr_annot__ beethoven::remote_ptr transformer_h_0_attn_qproj_weight_h5 PTR_FROM_OFFSET_H(0x2ce000L, 0x18000L);
__ptr_annot__ beethoven::remote_ptr transformer_h_0_attn_qproj_weight_h6 PTR_FROM_OFFSET_H(0x2e6000L, 0x18000L);
__ptr_annot__ beethoven::remote_ptr transformer_h_0_attn_qproj_weight_h7 PTR_FROM_OFFSET_H(0x2fe000L, 0x18000L);
__ptr_annot__ beethoven::remote_ptr transformer_h_0_attn_qproj_weight_h8 PTR_FROM_OFFSET_H(0x316000L, 0x18000L);
__ptr_annot__ beethoven::remote_ptr transformer_h_0_attn_qproj_weight_h9 PTR_FROM_OFFSET_H(0x32e000L, 0x18000L);
__ptr_annot__ beethoven::remote_ptr transformer_h_0_attn_qproj_weight_h10 PTR_FROM_OFFSET_H(0x346000L, 0x18000L);
__ptr_annot__ beethoven::remote_ptr transformer_h_0_attn_qproj_weight_h11 PTR_FROM_OFFSET_H(0x35e000L, 0x18000L);
__ptr_annot__ beethoven::remote_ptr transformer_h_0_attn_outproj_weight PTR_FROM_OFFSET_H(0x376000L, 0x120000L);
__ptr_annot__ beethoven::remote_ptr transformer_h_0_attn_outproj_bias PTR_FROM_OFFSET_H(0x496000L, 0x600L);
__ptr_annot__ beethoven::remote_ptr transformer_h_0_ln_2_gb PTR_FROM_OFFSET_H(0x497000L, 0xc00L);
__ptr_annot__ beethoven::remote_ptr transformer_h_0_mlp_cfc_weight PTR_FROM_OFFSET_H(0x498000L, 0x480000L);
__ptr_annot__ beethoven::remote_ptr transformer_h_0_mlp_cfc_bias PTR_FROM_OFFSET_H(0x918000L, 0x1800L);
__ptr_annot__ beethoven::remote_ptr transformer_h_0_mlp_cproj_weight PTR_FROM_OFFSET_H(0x91a000L, 0x480000L);
__ptr_annot__ beethoven::remote_ptr transformer_h_0_mlp_cproj_bias PTR_FROM_OFFSET_H(0xd9a000L, 0x600L);
__ptr_annot__ beethoven::remote_ptr transformer_h_1_ln_1_gb PTR_FROM_OFFSET_H(0xd9b000L, 0xc00L);
__ptr_annot__ beethoven::remote_ptr transformer_h_1_attn_kproj_weight_h0 PTR_FROM_OFFSET_H(0xd9c000L, 0x18000L);
__ptr_annot__ beethoven::remote_ptr transformer_h_1_attn_kproj_weight_h1 PTR_FROM_OFFSET_H(0xdb4000L, 0x18000L);
__ptr_annot__ beethoven::remote_ptr transformer_h_1_attn_kproj_weight_h2 PTR_FROM_OFFSET_H(0xdcc000L, 0x18000L);
__ptr_annot__ beethoven::remote_ptr transformer_h_1_attn_kproj_weight_h3 PTR_FROM_OFFSET_H(0xde4000L, 0x18000L);
__ptr_annot__ beethoven::remote_ptr transformer_h_1_attn_kproj_weight_h4 PTR_FROM_OFFSET_H(0xdfc000L, 0x18000L);
__ptr_annot__ beethoven::remote_ptr transformer_h_1_attn_kproj_weight_h5 PTR_FROM_OFFSET_H(0xe14000L, 0x18000L);
__ptr_annot__ beethoven::remote_ptr transformer_h_1_attn_kproj_weight_h6 PTR_FROM_OFFSET_H(0xe2c000L, 0x18000L);
__ptr_annot__ beethoven::remote_ptr transformer_h_1_attn_kproj_weight_h7 PTR_FROM_OFFSET_H(0xe44000L, 0x18000L);
__ptr_annot__ beethoven::remote_ptr transformer_h_1_attn_kproj_weight_h8 PTR_FROM_OFFSET_H(0xe5c000L, 0x18000L);
__ptr_annot__ beethoven::remote_ptr transformer_h_1_attn_kproj_weight_h9 PTR_FROM_OFFSET_H(0xe74000L, 0x18000L);
__ptr_annot__ beethoven::remote_ptr transformer_h_1_attn_kproj_weight_h10 PTR_FROM_OFFSET_H(0xe8c000L, 0x18000L);
__ptr_annot__ beethoven::remote_ptr transformer_h_1_attn_kproj_weight_h11 PTR_FROM_OFFSET_H(0xea4000L, 0x18000L);
__ptr_annot__ beethoven::remote_ptr transformer_h_1_attn_vproj_weight_h0 PTR_FROM_OFFSET_H(0xebc000L, 0x18000L);
__ptr_annot__ beethoven::remote_ptr transformer_h_1_attn_vproj_weight_h1 PTR_FROM_OFFSET_H(0xed4000L, 0x18000L);
__ptr_annot__ beethoven::remote_ptr transformer_h_1_attn_vproj_weight_h2 PTR_FROM_OFFSET_H(0xeec000L, 0x18000L);
__ptr_annot__ beethoven::remote_ptr transformer_h_1_attn_vproj_weight_h3 PTR_FROM_OFFSET_H(0xf04000L, 0x18000L);
__ptr_annot__ beethoven::remote_ptr transformer_h_1_attn_vproj_weight_h4 PTR_FROM_OFFSET_H(0xf1c000L, 0x18000L);
__ptr_annot__ beethoven::remote_ptr transformer_h_1_attn_vproj_weight_h5 PTR_FROM_OFFSET_H(0xf34000L, 0x18000L);
__ptr_annot__ beethoven::remote_ptr transformer_h_1_attn_vproj_weight_h6 PTR_FROM_OFFSET_H(0xf4c000L, 0x18000L);
__ptr_annot__ beethoven::remote_ptr transformer_h_1_attn_vproj_weight_h7 PTR_FROM_OFFSET_H(0xf64000L, 0x18000L);
__ptr_annot__ beethoven::remote_ptr transformer_h_1_attn_vproj_weight_h8 PTR_FROM_OFFSET_H(0xf7c000L, 0x18000L);
__ptr_annot__ beethoven::remote_ptr transformer_h_1_attn_vproj_weight_h9 PTR_FROM_OFFSET_H(0xf94000L, 0x18000L);
__ptr_annot__ beethoven::remote_ptr transformer_h_1_attn_vproj_weight_h10 PTR_FROM_OFFSET_H(0xfac000L, 0x18000L);
__ptr_annot__ beethoven::remote_ptr transformer_h_1_attn_vproj_weight_h11 PTR_FROM_OFFSET_H(0xfc4000L, 0x18000L);
__ptr_annot__ beethoven::remote_ptr transformer_h_1_attn_qproj_weight_h0 PTR_FROM_OFFSET_H(0xfdc000L, 0x18000L);
__ptr_annot__ beethoven::remote_ptr transformer_h_1_attn_qproj_weight_h1 PTR_FROM_OFFSET_H(0xff4000L, 0x18000L);
__ptr_annot__ beethoven::remote_ptr transformer_h_1_attn_qproj_weight_h2 PTR_FROM_OFFSET_H(0x100c000L, 0x18000L);
__ptr_annot__ beethoven::remote_ptr transformer_h_1_attn_qproj_weight_h3 PTR_FROM_OFFSET_H(0x1024000L, 0x18000L);
__ptr_annot__ beethoven::remote_ptr transformer_h_1_attn_qproj_weight_h4 PTR_FROM_OFFSET_H(0x103c000L, 0x18000L);
__ptr_annot__ beethoven::remote_ptr transformer_h_1_attn_qproj_weight_h5 PTR_FROM_OFFSET_H(0x1054000L, 0x18000L);
__ptr_annot__ beethoven::remote_ptr transformer_h_1_attn_qproj_weight_h6 PTR_FROM_OFFSET_H(0x106c000L, 0x18000L);
__ptr_annot__ beethoven::remote_ptr transformer_h_1_attn_qproj_weight_h7 PTR_FROM_OFFSET_H(0x1084000L, 0x18000L);
__ptr_annot__ beethoven::remote_ptr transformer_h_1_attn_qproj_weight_h8 PTR_FROM_OFFSET_H(0x109c000L, 0x18000L);
__ptr_annot__ beethoven::remote_ptr transformer_h_1_attn_qproj_weight_h9 PTR_FROM_OFFSET_H(0x10b4000L, 0x18000L);
__ptr_annot__ beethoven::remote_ptr transformer_h_1_attn_qproj_weight_h10 PTR_FROM_OFFSET_H(0x10cc000L, 0x18000L);
__ptr_annot__ beethoven::remote_ptr transformer_h_1_attn_qproj_weight_h11 PTR_FROM_OFFSET_H(0x10e4000L, 0x18000L);
__ptr_annot__ beethoven::remote_ptr transformer_h_1_attn_outproj_weight PTR_FROM_OFFSET_H(0x10fc000L, 0x120000L);
__ptr_annot__ beethoven::remote_ptr transformer_h_1_attn_outproj_bias PTR_FROM_OFFSET_H(0x121c000L, 0x600L);
__ptr_annot__ beethoven::remote_ptr transformer_h_1_ln_2_gb PTR_FROM_OFFSET_H(0x121d000L, 0xc00L);
__ptr_annot__ beethoven::remote_ptr transformer_h_1_mlp_cfc_weight PTR_FROM_OFFSET_H(0x121e000L, 0x480000L);
__ptr_annot__ beethoven::remote_ptr transformer_h_1_mlp_cfc_bias PTR_FROM_OFFSET_H(0x169e000L, 0x1800L);
__ptr_annot__ beethoven::remote_ptr transformer_h_1_mlp_cproj_weight PTR_FROM_OFFSET_H(0x16a0000L, 0x480000L);
__ptr_annot__ beethoven::remote_ptr transformer_h_1_mlp_cproj_bias PTR_FROM_OFFSET_H(0x1b20000L, 0x600L);
constexpr uint32_t allocator_base(0x1b21000);
#endif
