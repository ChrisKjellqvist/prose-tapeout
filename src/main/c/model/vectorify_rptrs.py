with open("../deploy/include/prose_rptr.h") as f:
  lns = f.readlines()

layers = 2
n_heads = 12

layer_str = ""
for i in range(layers):
  li = f"layers[{i}]"
  prefix = f"transformer_h_{i}"
  layer_str += f"""    {li}.ln1_wb = {prefix}_ln_1_gb;
    {li}.oproj_w = {prefix}_attn_outproj_weight;
    {li}.oproj_b = {prefix}_attn_outproj_bias;
    {li}.ln2_wb = {prefix}_ln_2_gb;
    {li}.mlp_fc_w = {prefix}_mlp_cfc_weight;
    {li}.mlp_fc_b = {prefix}_mlp_cfc_bias;
    {li}.mlp_proj_w = {prefix}_mlp_cproj_weight;
    {li}.causal_mask = {prefix}_attn_causal_mask;
    {li}.mlp_proj_b = {prefix}_mlp_cproj_bias;\n"""
  for j in range(n_heads):
    layer_str += f"""    {li}.proj_wgts[{j}].kproj = {prefix}_attn_kproj_weight_h{j};
    {li}.proj_wgts[{j}].vproj = {prefix}_attn_vproj_weight_h{j};
    {li}.proj_wgts[{j}].qproj = {prefix}_attn_qproj_weight_h{j};\n"""

with open("../deploy/include/prose_vec_rptr.h", 'w') as f:
  f.write(f"""#include "prose_rptr.h"
#include "prose_lib.h"
struct AllLayers {{
  TransformerLayer layers[{layers}];
  __constructor_annot__ ~AllLayers() {{}}
  __constructor_annot__ AllLayers() {{
{layer_str}
  }}
}};

extern const AllLayers all_layers;
""")

  