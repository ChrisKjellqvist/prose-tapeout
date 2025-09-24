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
__ptr_annot__ beethoven::remote_ptr transformer_h_1_aa_input PTR_FROM_OFFSET_H(0x5000L, 0x3000L);
__ptr_annot__ beethoven::remote_ptr transformer_h_1_attn_causal_mask PTR_FROM_OFFSET_H(0x8000L, 0x80L);
__ptr_annot__ beethoven::remote_ptr transformer_h_0_ln_1_gb PTR_FROM_OFFSET_H(0x9000L, 0xc00L);
__ptr_annot__ beethoven::remote_ptr transformer_h_0_attn_kproj_weight_h0 PTR_FROM_OFFSET_H(0xa000L, 0x18000L);
__ptr_annot__ beethoven::remote_ptr transformer_h_0_attn_kproj_weight_h1 PTR_FROM_OFFSET_H(0x22000L, 0x18000L);
__ptr_annot__ beethoven::remote_ptr transformer_h_0_attn_kproj_weight_h2 PTR_FROM_OFFSET_H(0x3a000L, 0x18000L);
__ptr_annot__ beethoven::remote_ptr transformer_h_0_attn_kproj_weight_h3 PTR_FROM_OFFSET_H(0x52000L, 0x18000L);
__ptr_annot__ beethoven::remote_ptr transformer_h_0_attn_kproj_weight_h4 PTR_FROM_OFFSET_H(0x6a000L, 0x18000L);
__ptr_annot__ beethoven::remote_ptr transformer_h_0_attn_kproj_weight_h5 PTR_FROM_OFFSET_H(0x82000L, 0x18000L);
__ptr_annot__ beethoven::remote_ptr transformer_h_0_attn_kproj_weight_h6 PTR_FROM_OFFSET_H(0x9a000L, 0x18000L);
__ptr_annot__ beethoven::remote_ptr transformer_h_0_attn_kproj_weight_h7 PTR_FROM_OFFSET_H(0xb2000L, 0x18000L);
__ptr_annot__ beethoven::remote_ptr transformer_h_0_attn_kproj_weight_h8 PTR_FROM_OFFSET_H(0xca000L, 0x18000L);
__ptr_annot__ beethoven::remote_ptr transformer_h_0_attn_kproj_weight_h9 PTR_FROM_OFFSET_H(0xe2000L, 0x18000L);
__ptr_annot__ beethoven::remote_ptr transformer_h_0_attn_kproj_weight_h10 PTR_FROM_OFFSET_H(0xfa000L, 0x18000L);
__ptr_annot__ beethoven::remote_ptr transformer_h_0_attn_kproj_weight_h11 PTR_FROM_OFFSET_H(0x112000L, 0x18000L);
__ptr_annot__ beethoven::remote_ptr transformer_h_0_attn_vproj_weight_h0 PTR_FROM_OFFSET_H(0x12a000L, 0x18000L);
__ptr_annot__ beethoven::remote_ptr transformer_h_0_attn_vproj_weight_h1 PTR_FROM_OFFSET_H(0x142000L, 0x18000L);
__ptr_annot__ beethoven::remote_ptr transformer_h_0_attn_vproj_weight_h2 PTR_FROM_OFFSET_H(0x15a000L, 0x18000L);
__ptr_annot__ beethoven::remote_ptr transformer_h_0_attn_vproj_weight_h3 PTR_FROM_OFFSET_H(0x172000L, 0x18000L);
__ptr_annot__ beethoven::remote_ptr transformer_h_0_attn_vproj_weight_h4 PTR_FROM_OFFSET_H(0x18a000L, 0x18000L);
__ptr_annot__ beethoven::remote_ptr transformer_h_0_attn_vproj_weight_h5 PTR_FROM_OFFSET_H(0x1a2000L, 0x18000L);
__ptr_annot__ beethoven::remote_ptr transformer_h_0_attn_vproj_weight_h6 PTR_FROM_OFFSET_H(0x1ba000L, 0x18000L);
__ptr_annot__ beethoven::remote_ptr transformer_h_0_attn_vproj_weight_h7 PTR_FROM_OFFSET_H(0x1d2000L, 0x18000L);
__ptr_annot__ beethoven::remote_ptr transformer_h_0_attn_vproj_weight_h8 PTR_FROM_OFFSET_H(0x1ea000L, 0x18000L);
__ptr_annot__ beethoven::remote_ptr transformer_h_0_attn_vproj_weight_h9 PTR_FROM_OFFSET_H(0x202000L, 0x18000L);
__ptr_annot__ beethoven::remote_ptr transformer_h_0_attn_vproj_weight_h10 PTR_FROM_OFFSET_H(0x21a000L, 0x18000L);
__ptr_annot__ beethoven::remote_ptr transformer_h_0_attn_vproj_weight_h11 PTR_FROM_OFFSET_H(0x232000L, 0x18000L);
__ptr_annot__ beethoven::remote_ptr transformer_h_0_attn_qproj_weight_h0 PTR_FROM_OFFSET_H(0x24a000L, 0x18000L);
__ptr_annot__ beethoven::remote_ptr transformer_h_0_attn_qproj_weight_h1 PTR_FROM_OFFSET_H(0x262000L, 0x18000L);
__ptr_annot__ beethoven::remote_ptr transformer_h_0_attn_qproj_weight_h2 PTR_FROM_OFFSET_H(0x27a000L, 0x18000L);
__ptr_annot__ beethoven::remote_ptr transformer_h_0_attn_qproj_weight_h3 PTR_FROM_OFFSET_H(0x292000L, 0x18000L);
__ptr_annot__ beethoven::remote_ptr transformer_h_0_attn_qproj_weight_h4 PTR_FROM_OFFSET_H(0x2aa000L, 0x18000L);
__ptr_annot__ beethoven::remote_ptr transformer_h_0_attn_qproj_weight_h5 PTR_FROM_OFFSET_H(0x2c2000L, 0x18000L);
__ptr_annot__ beethoven::remote_ptr transformer_h_0_attn_qproj_weight_h6 PTR_FROM_OFFSET_H(0x2da000L, 0x18000L);
__ptr_annot__ beethoven::remote_ptr transformer_h_0_attn_qproj_weight_h7 PTR_FROM_OFFSET_H(0x2f2000L, 0x18000L);
__ptr_annot__ beethoven::remote_ptr transformer_h_0_attn_qproj_weight_h8 PTR_FROM_OFFSET_H(0x30a000L, 0x18000L);
__ptr_annot__ beethoven::remote_ptr transformer_h_0_attn_qproj_weight_h9 PTR_FROM_OFFSET_H(0x322000L, 0x18000L);
__ptr_annot__ beethoven::remote_ptr transformer_h_0_attn_qproj_weight_h10 PTR_FROM_OFFSET_H(0x33a000L, 0x18000L);
__ptr_annot__ beethoven::remote_ptr transformer_h_0_attn_qproj_weight_h11 PTR_FROM_OFFSET_H(0x352000L, 0x18000L);
__ptr_annot__ beethoven::remote_ptr transformer_h_0_attn_outproj_weight PTR_FROM_OFFSET_H(0x36a000L, 0x120000L);
__ptr_annot__ beethoven::remote_ptr transformer_h_0_attn_outproj_bias PTR_FROM_OFFSET_H(0x48a000L, 0x600L);
__ptr_annot__ beethoven::remote_ptr transformer_h_0_ln_2_gb PTR_FROM_OFFSET_H(0x48b000L, 0xc00L);
__ptr_annot__ beethoven::remote_ptr transformer_h_0_mlp_cfc_weight PTR_FROM_OFFSET_H(0x48c000L, 0x480000L);
__ptr_annot__ beethoven::remote_ptr transformer_h_0_mlp_cfc_bias PTR_FROM_OFFSET_H(0x90c000L, 0x1800L);
__ptr_annot__ beethoven::remote_ptr transformer_h_0_mlp_cproj_weight PTR_FROM_OFFSET_H(0x90e000L, 0x480000L);
__ptr_annot__ beethoven::remote_ptr transformer_h_0_mlp_cproj_bias PTR_FROM_OFFSET_H(0xd8e000L, 0x600L);
__ptr_annot__ beethoven::remote_ptr transformer_h_1_ln_1_gb PTR_FROM_OFFSET_H(0xd8f000L, 0xc00L);
__ptr_annot__ beethoven::remote_ptr transformer_h_1_attn_kproj_weight_h0 PTR_FROM_OFFSET_H(0xd90000L, 0x18000L);
__ptr_annot__ beethoven::remote_ptr transformer_h_1_attn_kproj_weight_h1 PTR_FROM_OFFSET_H(0xda8000L, 0x18000L);
__ptr_annot__ beethoven::remote_ptr transformer_h_1_attn_kproj_weight_h2 PTR_FROM_OFFSET_H(0xdc0000L, 0x18000L);
__ptr_annot__ beethoven::remote_ptr transformer_h_1_attn_kproj_weight_h3 PTR_FROM_OFFSET_H(0xdd8000L, 0x18000L);
__ptr_annot__ beethoven::remote_ptr transformer_h_1_attn_kproj_weight_h4 PTR_FROM_OFFSET_H(0xdf0000L, 0x18000L);
__ptr_annot__ beethoven::remote_ptr transformer_h_1_attn_kproj_weight_h5 PTR_FROM_OFFSET_H(0xe08000L, 0x18000L);
__ptr_annot__ beethoven::remote_ptr transformer_h_1_attn_kproj_weight_h6 PTR_FROM_OFFSET_H(0xe20000L, 0x18000L);
__ptr_annot__ beethoven::remote_ptr transformer_h_1_attn_kproj_weight_h7 PTR_FROM_OFFSET_H(0xe38000L, 0x18000L);
__ptr_annot__ beethoven::remote_ptr transformer_h_1_attn_kproj_weight_h8 PTR_FROM_OFFSET_H(0xe50000L, 0x18000L);
__ptr_annot__ beethoven::remote_ptr transformer_h_1_attn_kproj_weight_h9 PTR_FROM_OFFSET_H(0xe68000L, 0x18000L);
__ptr_annot__ beethoven::remote_ptr transformer_h_1_attn_kproj_weight_h10 PTR_FROM_OFFSET_H(0xe80000L, 0x18000L);
__ptr_annot__ beethoven::remote_ptr transformer_h_1_attn_kproj_weight_h11 PTR_FROM_OFFSET_H(0xe98000L, 0x18000L);
__ptr_annot__ beethoven::remote_ptr transformer_h_1_attn_vproj_weight_h0 PTR_FROM_OFFSET_H(0xeb0000L, 0x18000L);
__ptr_annot__ beethoven::remote_ptr transformer_h_1_attn_vproj_weight_h1 PTR_FROM_OFFSET_H(0xec8000L, 0x18000L);
__ptr_annot__ beethoven::remote_ptr transformer_h_1_attn_vproj_weight_h2 PTR_FROM_OFFSET_H(0xee0000L, 0x18000L);
__ptr_annot__ beethoven::remote_ptr transformer_h_1_attn_vproj_weight_h3 PTR_FROM_OFFSET_H(0xef8000L, 0x18000L);
__ptr_annot__ beethoven::remote_ptr transformer_h_1_attn_vproj_weight_h4 PTR_FROM_OFFSET_H(0xf10000L, 0x18000L);
__ptr_annot__ beethoven::remote_ptr transformer_h_1_attn_vproj_weight_h5 PTR_FROM_OFFSET_H(0xf28000L, 0x18000L);
__ptr_annot__ beethoven::remote_ptr transformer_h_1_attn_vproj_weight_h6 PTR_FROM_OFFSET_H(0xf40000L, 0x18000L);
__ptr_annot__ beethoven::remote_ptr transformer_h_1_attn_vproj_weight_h7 PTR_FROM_OFFSET_H(0xf58000L, 0x18000L);
__ptr_annot__ beethoven::remote_ptr transformer_h_1_attn_vproj_weight_h8 PTR_FROM_OFFSET_H(0xf70000L, 0x18000L);
__ptr_annot__ beethoven::remote_ptr transformer_h_1_attn_vproj_weight_h9 PTR_FROM_OFFSET_H(0xf88000L, 0x18000L);
__ptr_annot__ beethoven::remote_ptr transformer_h_1_attn_vproj_weight_h10 PTR_FROM_OFFSET_H(0xfa0000L, 0x18000L);
__ptr_annot__ beethoven::remote_ptr transformer_h_1_attn_vproj_weight_h11 PTR_FROM_OFFSET_H(0xfb8000L, 0x18000L);
__ptr_annot__ beethoven::remote_ptr transformer_h_1_attn_qproj_weight_h0 PTR_FROM_OFFSET_H(0xfd0000L, 0x18000L);
__ptr_annot__ beethoven::remote_ptr transformer_h_1_attn_qproj_weight_h1 PTR_FROM_OFFSET_H(0xfe8000L, 0x18000L);
__ptr_annot__ beethoven::remote_ptr transformer_h_1_attn_qproj_weight_h2 PTR_FROM_OFFSET_H(0x1000000L, 0x18000L);
__ptr_annot__ beethoven::remote_ptr transformer_h_1_attn_qproj_weight_h3 PTR_FROM_OFFSET_H(0x1018000L, 0x18000L);
__ptr_annot__ beethoven::remote_ptr transformer_h_1_attn_qproj_weight_h4 PTR_FROM_OFFSET_H(0x1030000L, 0x18000L);
__ptr_annot__ beethoven::remote_ptr transformer_h_1_attn_qproj_weight_h5 PTR_FROM_OFFSET_H(0x1048000L, 0x18000L);
__ptr_annot__ beethoven::remote_ptr transformer_h_1_attn_qproj_weight_h6 PTR_FROM_OFFSET_H(0x1060000L, 0x18000L);
__ptr_annot__ beethoven::remote_ptr transformer_h_1_attn_qproj_weight_h7 PTR_FROM_OFFSET_H(0x1078000L, 0x18000L);
__ptr_annot__ beethoven::remote_ptr transformer_h_1_attn_qproj_weight_h8 PTR_FROM_OFFSET_H(0x1090000L, 0x18000L);
__ptr_annot__ beethoven::remote_ptr transformer_h_1_attn_qproj_weight_h9 PTR_FROM_OFFSET_H(0x10a8000L, 0x18000L);
__ptr_annot__ beethoven::remote_ptr transformer_h_1_attn_qproj_weight_h10 PTR_FROM_OFFSET_H(0x10c0000L, 0x18000L);
__ptr_annot__ beethoven::remote_ptr transformer_h_1_attn_qproj_weight_h11 PTR_FROM_OFFSET_H(0x10d8000L, 0x18000L);
__ptr_annot__ beethoven::remote_ptr transformer_h_1_attn_outproj_weight PTR_FROM_OFFSET_H(0x10f0000L, 0x120000L);
__ptr_annot__ beethoven::remote_ptr transformer_h_1_attn_outproj_bias PTR_FROM_OFFSET_H(0x1210000L, 0x600L);
__ptr_annot__ beethoven::remote_ptr transformer_h_1_ln_2_gb PTR_FROM_OFFSET_H(0x1211000L, 0xc00L);
__ptr_annot__ beethoven::remote_ptr transformer_h_1_mlp_cfc_weight PTR_FROM_OFFSET_H(0x1212000L, 0x480000L);
__ptr_annot__ beethoven::remote_ptr transformer_h_1_mlp_cfc_bias PTR_FROM_OFFSET_H(0x1692000L, 0x1800L);
__ptr_annot__ beethoven::remote_ptr transformer_h_1_mlp_cproj_weight PTR_FROM_OFFSET_H(0x1694000L, 0x480000L);
__ptr_annot__ beethoven::remote_ptr transformer_h_1_mlp_cproj_bias PTR_FROM_OFFSET_H(0x1b14000L, 0x600L);
constexpr uint32_t allocator_base(0x1b15000);
#endif
