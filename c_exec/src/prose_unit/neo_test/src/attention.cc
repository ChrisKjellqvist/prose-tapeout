//
// Created by Entropy Xu on 9/10/24.
//

#ifdef USE_TORCH
#include "torch/torch.h"
#endif
#include <cstring>
#include <float_wrapper.h>
#include <fstream>
#include <interchange_format.h>
#include <iostream>
#include <torch_util.h>
#include <util.h>

#include "prose_golden.h"


int main() {
  const int batch_size = 1;
  const int seq_len = 4;
  const int embed_dim = 768;
  const int num_heads = 12;

#ifdef USE_TORCH
  std::cout << (get_global_checkpoint_dir() + "/attn.k_proj.weight.pt")
            << std::endl;
  auto k_proj_w =
      load_tensor(get_global_checkpoint_dir() + "/attn.k_proj.weight.pt");
  auto v_proj_w =
      load_tensor(get_global_checkpoint_dir() + "/attn.v_proj.weight.pt");
  auto q_proj_w =
      load_tensor(get_global_checkpoint_dir() + "/attn.q_proj.weight.pt");
  auto o_proj_w =
      load_tensor(get_global_checkpoint_dir() + "/attn.o_proj.weight.pt");
  auto o_proj_b =
      load_tensor(get_global_checkpoint_dir() + "/attn.o_proj.bias.pt");
  auto causal_mask =
      load_tensor(get_global_checkpoint_dir() + "/attn.causal_mask.pt");
  auto golden_input =
      load_tensor(get_global_checkpoint_dir() + "/attn.input.pt");
  auto golden_output =
      load_tensor(get_global_checkpoint_dir() + "/attn.output.pt");

  assert(golden_input.size(0) == batch_size);
  assert(seq_len == golden_input.size(1));
  assert(embed_dim == golden_input.size(2));
  auto attn_output =
      torch::zeros({batch_size, seq_len, embed_dim}, torch::kFloat32);

  prose_isa_golden::prose_multi_head_attention(
      golden_input.data_ptr<float>(), q_proj_w.data_ptr<float>(),
      k_proj_w.data_ptr<float>(), v_proj_w.data_ptr<float>(),
      o_proj_w.data_ptr<float>(), o_proj_b.data_ptr<float>(),
      causal_mask.data_ptr<float>(), batch_size, seq_len, embed_dim, num_heads,
      attn_output.data_ptr<float>());
  std::cout << "golden" << std::endl;
  float biggest_diff = 0;
  auto a1 = golden_output.contiguous();
  auto a2 = attn_output.contiguous();
  for (int k = 0; k < seq_len; ++k) {
    for (int i = 0; i < embed_dim; ++i) {
      auto gold = a1.data_ptr<float>()[i];
      auto asdf = a2.data_ptr<float>()[i];
      auto diff = std::abs((gold - asdf) / gold) * 100;
      if (diff > biggest_diff && std::abs(gold - asdf) > 0.001) {
        biggest_diff = diff;
      }
      if (diff > 0.1 && std::abs(gold - asdf) > 0.001) {
        // printf("%0.4f\t%0.4f\t%0.4f%%, %d %d\n", gold, asdf, diff, k, i);
      }
    }
  }
  std::cout << "biggest diff: " << biggest_diff << std::endl;
  std::cout << std::endl;
  auto diff = attn_output - golden_output;
  std::cout << diff.sizes() << std::endl;
  std::cout << "Diff Matrix: " << std::endl;
  print_small_tensor(diff[0]);

  //  auto pct_diff = (diff / golden_output * 100).abs().max().item<float>();
  std::cout << "Max Pct diff: " << biggest_diff << "%" << std::endl;
  // assert(biggest_diff < 0.1);
  std::cout << "Test passed!" << std::endl;

  // since the test passed, write the arrays to the interchange text format so
  // we can run off-chip without needing pytorch
  std::vector<int64_t> size;

#define WRITE_TO_F(tensor, fname)                                              \
  interchange_format(tensor).write_floats_to_file(fname);

  WRITE_TO_F(golden_input, get_text_checkpoint_dir() + "/input.float")

  // pre-split the heads for multi-head self attention
  auto qsplit = multi_head_torch_tensor_to_flt_array(q_proj_w, num_heads, 1);
  auto ksplit = multi_head_torch_tensor_to_flt_array(k_proj_w, num_heads, 1);
  auto vsplit = multi_head_torch_tensor_to_flt_array(v_proj_w, num_heads, 1);
  auto osplit =
      multi_head_torch_tensor_to_flt_array(o_proj_w.t(), num_heads, 0);
  for (int h = 0; h < num_heads; ++h) {
    auto q2 = qsplit[h];
    auto k2 = ksplit[h];
    auto v2 = vsplit[h];
    auto o2 = osplit[h];
    // this turns the qkv matrices to 64x768
    // and the o matrix to 768x64
    WRITE_TO_F(q2,
               get_text_checkpoint_dir() + "/h" + std::to_string(h) +
                   ".q_proj_w.float");
    WRITE_TO_F(k2,
               get_text_checkpoint_dir() + "/h" + std::to_string(h) +
                   ".k_proj_w.float");
    WRITE_TO_F(v2,
               get_text_checkpoint_dir() + "/h" + std::to_string(h) +
                   ".v_proj_w.float");
    WRITE_TO_F(o2,
               get_text_checkpoint_dir() + "/h" + std::to_string(h) +
                   ".o_proj_w.float");
  }
  WRITE_TO_F(o_proj_b, get_text_checkpoint_dir() + "/o_proj_b.float");
  WRITE_TO_F(causal_mask, get_text_checkpoint_dir() + "/causal_mask.float");
  WRITE_TO_F(attn_output, get_text_checkpoint_dir() + "/golden_output.float");
#endif

#ifdef TEST_PROSE
  std::cout << "starting hw" << std::endl;
  float **qarray, **karray, **varray, **oarray;

  qarray = new float*[num_heads];
  karray = new float*[num_heads];
  varray = new float*[num_heads];
  oarray = new float*[num_heads];

  for (int h = 0; h < num_heads; ++h) {
    auto if_qproj = interchange_format::from_float_file(
        get_text_checkpoint_dir() + "/h" + std::to_string(h) +
            ".q_proj_w.float",
        {embed_dim / num_heads, embed_dim});
    auto if_kproj = interchange_format::from_float_file(
        get_text_checkpoint_dir() + "/h" + std::to_string(h) +
            ".k_proj_w.float",
        {embed_dim / num_heads, embed_dim});
    auto if_vproj = interchange_format::from_float_file(
        get_text_checkpoint_dir() + "/h" + std::to_string(h) +
            ".v_proj_w.float",
        {embed_dim / num_heads, embed_dim});
    auto if_oprojw = interchange_format::from_float_file(
        get_text_checkpoint_dir() + "/h" + std::to_string(h) +
            ".o_proj_w.float",
        {embed_dim, embed_dim / num_heads});
    qarray[h] = if_qproj.data;
    karray[h] = if_kproj.data;
    varray[h] = if_vproj.data;
    oarray[h] = if_oprojw.data;
  }

  auto if_oprojb = interchange_format::from_float_file(
      get_text_checkpoint_dir() + "/o_proj_b.float", {embed_dim});
  auto if_causal = interchange_format::from_float_file(
      get_text_checkpoint_dir() + "/causal_mask.float", {1, seq_len, seq_len});
  auto if_input = interchange_format::from_float_file(
      get_text_checkpoint_dir() + "/input.float",
      {batch_size, seq_len, embed_dim});
  auto if_gout = interchange_format::from_float_file(
      get_text_checkpoint_dir() + "/golden_output.float",
      {batch_size, seq_len, embed_dim});

  float *output = new float[batch_size * seq_len * embed_dim],
        *output_swap = new float[batch_size * seq_len * embed_dim];
  memset(output, 0, sizeof(float) * seq_len * embed_dim);
  prose_float_wrapper::prose_multi_head_attention(
      if_input.data, qarray, karray, varray, oarray, if_oprojb.data,
      if_causal.data, batch_size, seq_len, embed_dim, num_heads, output);



  auto gold = if_gout.data;
  float max_diff = 0;
  float max_pct_diff = 0;
  float max_f = -10000;
  for (int i = 0; i < seq_len * embed_dim; ++i) {
    float diff = std::abs(output[i] - gold[i]);
    printf("%d\t%0.4f\t%0.4f\n", i, output[i], gold[i]);
    if (diff > max_diff) {
      max_diff = diff;
      std::cout << "max diff it(" << i << "): " << diff << " on gold(" << gold[i] << ") obs(" << output[i] << ")" <<  std::endl;
    }
    float pct_diff = diff / std::abs(gold[i]) * 100;
    if (pct_diff > max_pct_diff) {
      max_pct_diff = pct_diff;
      std::cout << "max pct diff it(" << i << "): " << pct_diff << " on gold(" << gold[i] << ") obs(" << output[i] << ")" << std::endl;
    }
    if (std::abs(output[i]) > max_f)
      max_f = std::abs(output[i]);
  }
  std::cout << "Max diff: " << max_diff << std::endl;
  std::cout << "Max pct diff: " << max_pct_diff << "%" << std::endl;
  std::cout << "Max item: " << max_f << std::endl;

  handle.shutdown();
#endif
  return 0;
}
