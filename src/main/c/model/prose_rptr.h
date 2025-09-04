#ifndef PROSE_RPTR_H
#define PROSE_RPTR_H
#include <cstdint>
#include <beethoven/allocator/alloc_baremetal.h>
const beethoven::remote_ptr DEBUG(0x0L);// 1.00 KB, Total: 00 MB
const beethoven::remote_ptr transformer_h_0_ln_1_weight(0x1000L);// 1.50 KB, Total: 00 MB
const beethoven::remote_ptr transformer_h_0_ln_1_bias(0x2000L);// 1.50 KB, Total: 00 MB
const beethoven::remote_ptr transformer_h_0_attn_kproj_weight(0x3000L);// 1.12 MB, Total: 00 MB
const beethoven::remote_ptr transformer_h_0_attn_vproj_weight(0x123000L);// 1.12 MB, Total: 01 MB
const beethoven::remote_ptr transformer_h_0_attn_qproj_weight(0x243000L);// 1.12 MB, Total: 02 MB
const beethoven::remote_ptr transformer_h_0_attn_outproj_weight(0x363000L);// 1.12 MB, Total: 03 MB
const beethoven::remote_ptr transformer_h_0_attn_outproj_bias(0x483000L);// 1.50 KB, Total: 05 MB
const beethoven::remote_ptr transformer_h_0_ln_2_weight(0x484000L);// 1.50 KB, Total: 05 MB
const beethoven::remote_ptr transformer_h_0_ln_2_bias(0x485000L);// 1.50 KB, Total: 05 MB
const beethoven::remote_ptr transformer_h_0_mlp_cfc_weight(0x486000L);// 4.50 MB, Total: 05 MB
const beethoven::remote_ptr transformer_h_0_mlp_cfc_bias(0x906000L);// 6.00 KB, Total: 09 MB
const beethoven::remote_ptr transformer_h_0_mlp_cproj_weight(0x908000L);// 4.50 MB, Total: 09 MB
const beethoven::remote_ptr transformer_h_0_mlp_cproj_bias(0xd88000L);// 1.50 KB, Total: 13.54 MB
const uint32_t allocator_base = 0xd89000;
#endif
