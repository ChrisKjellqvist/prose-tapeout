//
// Created by Entropy Xu on 9/10/24.
//

#include <cstring>
#include <float_wrapper.h>
#include <fstream>
#include <interchange_format.h>
#include <iostream>
#include <torch_util.h>
#include <util.h>

#include "prose_golden.h"


#ifdef USE_TORCH

#include "torch/torch.h"
void print_a_bit(torch::Tensor a) {
  int l = a.sizes()[2];
  auto *p = a.data_ptr<float>();
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) {
      printf("%0.4f\t", p[i * l + j]);
    }
    printf("\n");
  }
}

#endif

void print_a_bit(float *f) {
  int l = 768;
  for (int i = 0; i < 2; ++i) {
    for (int j = 0; j < 2; ++j) {
      printf("%0.4f\t", f[i * l + j]);
    }
    printf("\n");
  }
}
const int batch_size = 1;
const int seq_len = 4;
const int embed_dim = 768;
const int num_heads = 12;
const int layer_num = 0;


float compare(float *a1, float *a2) {
  float biggest_diff = 0;
  for (int k = 0; k < seq_len; ++k) {
    for (int i = 0; i < embed_dim; ++i) {
      auto gold = a1[i];
      auto asdf = a2[i];
      auto diff = std::abs(gold - asdf);
      if (diff > biggest_diff) {
        biggest_diff = diff;
        printf("%0.4f\t%0.4f\t%0.4f%%, %d %d\n", gold, asdf, 100.0*std::abs(diff/gold), k, i);
      }
    }
  }
  return biggest_diff;
}


int main() {
#define g_fname(mat, layer) (get_global_checkpoint_dir() + "/transformer.h." + std::to_string(layer) +  "." #mat ".pt")

#ifdef USE_TORCH

  //  std::cout << (get_global_checkpoint_dir() + "/attn.k_proj.weight.pt") << std::endl;
  std::cout << g_fname(attn.attention.k_proj.weight, layer_num) << std::endl;
  auto k_proj_w = load_tensor(g_fname(attn.attention.k_proj.weight, layer_num));
  auto v_proj_w = load_tensor(g_fname(attn.attention.v_proj.weight, layer_num));
  auto q_proj_w = load_tensor(g_fname(attn.attention.q_proj.weight, layer_num));
  auto o_proj_w = load_tensor(g_fname(attn.attention.out_proj.weight, layer_num));
  auto o_proj_b = load_tensor(g_fname(attn.attention.out_proj.bias, layer_num));
  auto causal_mask = load_tensor(g_fname(attn.attention.causal_mask, layer_num));
  auto golden_input = load_tensor(get_global_checkpoint_dir() + "/input.pt");
  auto golden_output = load_tensor(g_fname(output, layer_num));
  auto ln1_w = load_tensor(g_fname(ln_1.weight, layer_num));
  auto ln1_b = load_tensor(g_fname(ln_1.bias, layer_num));
  auto ln2_w = load_tensor(g_fname(ln_2.weight, layer_num));
  auto ln2_b = load_tensor(g_fname(ln_2.bias, layer_num));
  auto fc_w = load_tensor(g_fname(mlp.c_fc.weight, layer_num)).t().contiguous();
  auto fc_b = load_tensor(g_fname(mlp.c_fc.bias, layer_num));
  auto c_proj_w = load_tensor(g_fname(mlp.c_proj.weight, layer_num)).t().contiguous();
  std::cout << "FW: " << c_proj_w.sizes() << std::endl;
  auto c_proj_b = load_tensor(g_fname(mlp.c_proj.bias, layer_num));

  assert(golden_input.size(0) == batch_size);
  assert(seq_len == golden_input.size(1));
  assert(embed_dim == golden_input.size(2));
  auto attn_output = torch::zeros({batch_size, seq_len, embed_dim}, torch::kFloat32);


#define tfl(x) x.data_ptr<float>()
  prose_isa_golden::decoder_layer(tfl(golden_input),
                                  tfl(ln1_w),
                                  tfl(ln1_b),
                                  tfl(ln2_w),
                                  tfl(ln2_b),
                                  tfl(q_proj_w),
                                  tfl(k_proj_w),
                                  tfl(v_proj_w),
                                  tfl(o_proj_w),
                                  tfl(o_proj_b),
                                  tfl(causal_mask),
                                  tfl(fc_w),
                                  tfl(fc_b),
                                  tfl(c_proj_w),
                                  tfl(c_proj_b),
                                  batch_size,
                                  seq_len,
                                  embed_dim,
                                  12,
                                  attn_output.data_ptr<float>()
  );

  print_a_bit(attn_output);

  std::cout << "golden" << std::endl;
  auto a1 = golden_output.contiguous();
  auto a2 = attn_output.contiguous();
  auto biggest_diff = compare(a1.data_ptr<float>(), a2.data_ptr<float>());
  std::cout << "biggest diff: " << biggest_diff << std::endl;
  auto diff = attn_output - golden_output;

  // since the test passed, write the arrays to the interchange text format so
  // we can run off-chip without needing pytorch
  std::vector<int64_t> size;

#define WRITE_TO_F(tensor, fname)                                              \
interchange_format(tensor).write_floats_to_file(fname)

  WRITE_TO_F(golden_input, get_text_checkpoint_dir() + "/input.float");

  // pre-split the heads for multi-head self attention
  auto qsplit = multi_head_torch_tensor_to_flt_array(q_proj_w.t(), num_heads, 1);
  auto ksplit = multi_head_torch_tensor_to_flt_array(k_proj_w.t(), num_heads, 1);
  auto vsplit = multi_head_torch_tensor_to_flt_array(v_proj_w.t(), num_heads, 1);
  auto osplit = multi_head_torch_tensor_to_flt_array(o_proj_w.t(), num_heads, 0);
  for (int h = 0; h < num_heads; ++h) {
    auto q2 = qsplit[h];
    auto k2 = ksplit[h];
    auto v2 = vsplit[h];
    auto o2 = osplit[h];
    // this turns the qkv matrices to 64x768
    // and the o matrix to 768x64
#define t_fname_no_head(mat, layer) (get_text_checkpoint_dir() + "/transformer.h." + std::to_string(layer) +  "." #mat ".float")
#define t_fname(mat, layer, head) (get_text_checkpoint_dir() + "/transformer.h." + std::to_string(layer) +  "." + std::to_string(head) + "." + #mat ".float")

    WRITE_TO_F(q2, t_fname(attn.attention.q_proj.weight, layer_num, h));
    WRITE_TO_F(k2, t_fname(attn.attention.k_proj.weight, layer_num, h));
    WRITE_TO_F(v2, t_fname(attn.attention.v_proj.weight, layer_num, h));
    WRITE_TO_F(o2, t_fname(attn.attention.out_proj.weight, layer_num, h));
  }
  WRITE_TO_F(o_proj_b, t_fname_no_head(attn.attention.out_proj.bias, layer_num));
  WRITE_TO_F(causal_mask, t_fname_no_head(attn.attention.causal_mask, layer_num));
  WRITE_TO_F(attn_output, t_fname_no_head(output, layer_num));
  WRITE_TO_F(ln1_w, t_fname_no_head(ln_1.weight, layer_num));
  WRITE_TO_F(ln1_b, t_fname_no_head(ln_1.bias, layer_num));
  WRITE_TO_F(ln2_w, t_fname_no_head(ln_2.weight, layer_num));
  WRITE_TO_F(ln2_b, t_fname_no_head(ln_2.bias, layer_num));
  WRITE_TO_F(fc_w, t_fname_no_head(mlp.c_fc.weight, layer_num));
  WRITE_TO_F(fc_b, t_fname_no_head(mlp.c_fc.bias, layer_num));
  WRITE_TO_F(c_proj_w, t_fname_no_head(mlp.c_proj.weight, layer_num));
  WRITE_TO_F(c_proj_b, t_fname_no_head(mlp.c_proj.bias, layer_num));

#endif

#ifdef TEST_PROSE
  std::cout << "starting hw" << std::endl;
  float **qarray, **karray, **varray, **oarray;

  qarray = new float*[num_heads];
  karray = new float*[num_heads];
  varray = new float*[num_heads];
  oarray = new float*[num_heads];
#define get_text_head(name, layer, head, dims) interchange_format::from_float_file(\
get_text_checkpoint_dir() + "/transformer.h." + std::to_string(layer) + "." + std::to_string(head) + ("." #name ".float"), dims)
  std::vector<int64_t> head_dims = {embed_dim / num_heads, embed_dim};
  for (int h = 0; h < num_heads; ++h) {

    auto if_qproj = get_text_head(attn.attention.q_proj.weight, layer_num, h, head_dims);
    auto if_kproj = get_text_head(attn.attention.k_proj.weight, layer_num, h, head_dims);
    auto if_vproj = get_text_head(attn.attention.v_proj.weight, layer_num, h, head_dims);
    auto if_oproj = get_text_head(attn.attention.out_proj.weight, layer_num, h, head_dims);
    qarray[h] = if_qproj.data;
    karray[h] = if_kproj.data;
    varray[h] = if_vproj.data;
    oarray[h] = if_oproj.data;
  }

#define get_text(name, layer, ...) interchange_format::from_float_file(\
get_text_checkpoint_dir() + "/transformer.h." + std::to_string(layer) + "." + #name + ".float", {__VA_ARGS__})

  std::vector<int64_t> embed_dims = {embed_dim, embed_dim};
  std::vector<int64_t> embed_vec = {embed_dim};
  std::vector<int64_t> causal_dim = {1, seq_len, seq_len};
  std::vector<int64_t> input_dim = {1, seq_len, embed_dim};
  auto if_oprojb = get_text(attn.attention.out_proj.bias, layer_num, embed_vec);
  auto if_causal = get_text(attn.attention.causal_mask, layer_num, {1, seq_len, seq_len});
  auto if_input = interchange_format::from_float_file(
    get_text_checkpoint_dir() + "/input.float", {batch_size, seq_len, embed_dim});
  auto if_gout = get_text(output, layer_num, input_dim);
  auto if_ln1_w = get_text(ln_1.weight, layer_num, embed_vec);
  auto if_ln1_b = get_text(ln_1.bias, layer_num, embed_vec);
  auto if_ln2_w = get_text(ln_2.weight, layer_num, embed_vec);
  auto if_ln2_b = get_text(ln_2.bias, layer_num, embed_vec);

  auto if_mlp_fc_wgt = get_text(mlp.c_fc.weight, layer_num, {embed_dim, 3072});
  auto if_mlp_fc_bias = get_text(mlp.c_fc.bias, layer_num, {3072});
  auto if_mlp_proj_wgt = get_text(mlp.c_proj.weight, layer_num, {3072, embed_dim});
  auto if_mlp_proj_bias = get_text(mlp.c_proj.bias, layer_num, {embed_dim});


  float *output = new float[batch_size * seq_len * embed_dim];
  memset(output, 0, sizeof(float) * seq_len * embed_dim);

  prose_float_wrapper::decoder_layer(if_input.data,
    if_ln1_w.data,
    if_ln1_b.data,
    if_ln2_w.data,
    if_ln2_b.data,
    qarray,
    karray,
    varray,
    oarray,
    if_oprojb.data,
    if_causal.data,
    if_mlp_fc_wgt.data,
    if_mlp_fc_bias.data,
    if_mlp_proj_wgt.data,
    if_mlp_proj_bias.data,
    batch_size,
    seq_len,
    embed_dim,
    num_heads,
    output);

  auto gold = if_gout.data;
  auto big_diff = compare(gold, output);
  std::cout << "biggest diff: " << big_diff << std::endl;


#endif
#ifdef USE_TORCH
  handle.shutdown();
#endif
  return 0;
}
