
#include "prose_golden.h"
#include "torch/torch.h"
#include "util.h"
#include <fstream>
#include <iostream>
#include <random>
#include <torch_util.h>

std::string model_prefix;

struct GPTNeoBlockParams {
  torch::Tensor attn_k_proj_weight;
  torch::Tensor attn_q_proj_weight;
  torch::Tensor attn_v_proj_weight;
  torch::Tensor attn_out_proj_weight;
  torch::Tensor attn_out_proj_bias;
  torch::Tensor ln_1_weight;
  torch::Tensor ln_1_bias;
  torch::Tensor ln_2_weight;
  torch::Tensor ln_2_bias;
  torch::Tensor mlp_c_fc_weight;
  torch::Tensor mlp_c_fc_bias;
  torch::Tensor mlp_c_proj_weight;
  torch::Tensor mlp_c_proj_bias;
};

std::vector<GPTNeoBlockParams> load_gpt_neo_params() {
  std::vector<GPTNeoBlockParams> params;

  for (int i = 0; i < 12; ++i) {  // Assuming 12 layers based on the file names
    GPTNeoBlockParams layer_params;

    layer_params.attn_k_proj_weight = load_tensor(
            model_prefix + "model_h." + std::to_string(i) + ".attn.attention.k_proj.weight.pt");
    layer_params.attn_q_proj_weight = load_tensor(
            model_prefix + "model_h." + std::to_string(i) + ".attn.attention.q_proj.weight.pt");
    layer_params.attn_v_proj_weight = load_tensor(
            model_prefix + "model_h." + std::to_string(i) + ".attn.attention.v_proj.weight.pt");
    layer_params.attn_out_proj_weight = load_tensor(
            model_prefix + "model_h." + std::to_string(i) + ".attn.attention.out_proj.weight.pt");
    layer_params.attn_out_proj_bias = load_tensor(
            model_prefix + "model_h." + std::to_string(i) + ".attn.attention.out_proj.bias.pt");

    layer_params.ln_1_weight = load_tensor(model_prefix + "model_h." + std::to_string(i) + ".ln_1.weight.pt");
    layer_params.ln_1_bias = load_tensor(model_prefix + "model_h." + std::to_string(i) + ".ln_1.bias.pt");
    layer_params.ln_2_weight = load_tensor(model_prefix + "model_h." + std::to_string(i) + ".ln_2.weight.pt");
    layer_params.ln_2_bias = load_tensor(model_prefix + "model_h." + std::to_string(i) + ".ln_2.bias.pt");

    layer_params.mlp_c_fc_weight = load_tensor(model_prefix + "model_h." + std::to_string(i) + ".mlp.c_fc.weight.pt");
    layer_params.mlp_c_fc_bias = load_tensor(model_prefix + "model_h." + std::to_string(i) + ".mlp.c_fc.bias.pt");
    layer_params.mlp_c_proj_weight = load_tensor(model_prefix + "model_h." + std::to_string(i) + ".mlp.c_proj.weight.pt");
    layer_params.mlp_c_proj_bias = load_tensor(model_prefix + "model_h." + std::to_string(i) + ".mlp.c_proj.bias.pt");

    params.push_back(layer_params);
  }

  return params;
}

torch::Tensor
gpt_neo_decoder_layer(const GPTNeoBlockParams &params, const torch::Tensor &input, const torch::Tensor &causal_mask) {

  auto batch_size = input.size(0);
  auto input_length = input.size(1);
  auto embedding_size = input.size(2);
  uint16_t num_heads = 12;

  // Layer norm 1
  auto after_ln_1 = torch::zeros_like(input);
  prose_isa_golden::prose_layer_norm(input.data_ptr<float>(),
                                     params.ln_1_weight.data_ptr<float>(),
                                     params.ln_1_bias.data_ptr<float>(),
                                     batch_size,
                                     input_length,
                                     embedding_size,
                                     after_ln_1.data_ptr<float>());

  // Multi-head attention
  auto after_attn = torch::zeros_like(input);
  prose_isa_golden::prose_multi_head_attention(after_ln_1.data_ptr<float>(),
                                               params.attn_q_proj_weight.data_ptr<float>(),
                                               params.attn_k_proj_weight.data_ptr<float>(),
                                               params.attn_v_proj_weight.data_ptr<float>(),
                                               params.attn_out_proj_weight.data_ptr<float>(),
                                               params.attn_out_proj_bias.data_ptr<float>(),
                                               causal_mask.data_ptr<float>(),
                                               batch_size,
                                               input_length,
                                               embedding_size,
                                               num_heads,
                                               after_attn.data_ptr<float>());

  // Residual connection and layer norm 2
  auto after_residual_1 = after_attn + input;
  auto after_ln_2 = torch::zeros_like(input);
  prose_isa_golden::prose_layer_norm(after_residual_1.data_ptr<float>(),
                                     params.ln_2_weight.data_ptr<float>(),
                                     params.ln_2_bias.data_ptr<float>(),
                                     batch_size,
                                     input_length,
                                     embedding_size,
                                     after_ln_2.data_ptr<float>());

  // MLP
  auto c_fc_out = torch::zeros({batch_size, input_length, embedding_size * 4}, torch::kFloat32);
  prose_isa_golden::prose_g_matmul(after_ln_2.data_ptr<float>(),
                                   params.mlp_c_fc_weight.data_ptr<float>(),
                                   nullptr,
                                   params.mlp_c_fc_bias.data_ptr<float>(),
                                   c_fc_out.data_ptr<float>(),
                                   batch_size, input_length, embedding_size, embedding_size * 4);

  auto after_mlp = torch::zeros_like(input);
  prose_isa_golden::prose_m_matmul(c_fc_out.data_ptr<float>(),
                                   params.mlp_c_proj_weight.data_ptr<float>(),
                                   after_mlp.data_ptr<float>(),
                                   params.mlp_c_proj_bias.data_ptr<float>(),
                                   batch_size, input_length, embedding_size * 4, embedding_size, false, nullptr);

  // Final residual connection
  auto output = after_mlp + after_residual_1;

  return output;
}

torch::Tensor gpt_neo_model(torch::Tensor &inputs_embeds,
                            std::vector<GPTNeoBlockParams> &params,
                            torch::Tensor &causal_mask,
                            torch::Tensor &final_ln_weight,
                            torch::Tensor &final_ln_bias) {
  int batch_size = inputs_embeds.size(0);
  int input_length = inputs_embeds.size(1);
  int embedding_size = inputs_embeds.size(2);
  int num_layers = params.size();

  auto hidden_states = inputs_embeds;

  // Loop through all layers
  for (int i = 0; i < num_layers; ++i) {
    hidden_states = gpt_neo_decoder_layer(params[i], hidden_states, causal_mask);
    // print the shape of hidden_states
    std::cout << "Hidden states shape: " << hidden_states.sizes() << std::endl;
  }

  // Final layer norm
  auto output = torch::zeros_like(hidden_states);
  prose_isa_golden::prose_layer_norm(hidden_states.data_ptr<float>(),
                                     final_ln_weight.data_ptr<float>(),
                                     final_ln_bias.data_ptr<float>(),
                                     batch_size,
                                     input_length,
                                     embedding_size,
                                     output.data_ptr<float>());

  return output;
}


int main(int argc, char ** argv) {
  if (argc != 2) {
    std::cerr << "Usage: " << argv[0] << " <model_prefix>" << std::endl;
    return 1;
  }
  model_prefix = std::string(argv[1]) + "/";
  auto params = load_gpt_neo_params();
  auto causal_mask = load_tensor(model_prefix + "attn.causal_mask.pt");
  auto sample_input = load_tensor(model_prefix + "model_input_block.pt");
  auto golden_output = load_tensor(model_prefix + "model_output_block.pt");
  auto ln_f_bias = load_tensor(model_prefix + "model_ln_f.bias.pt");
  auto ln_f_weight = load_tensor(model_prefix + "model_ln_f.weight.pt");
  // print the shape of golden input and output
  std::cout << "Golden input shape: " << sample_input.sizes() << std::endl;
  std::cout << "Golden output shape: " << golden_output.sizes() << std::endl;

  auto output = gpt_neo_model(sample_input, params, causal_mask, ln_f_weight, ln_f_bias);

  // print the output shape
  std::cout << "Output shape: " << output.sizes() << std::endl;

  auto diff = output - golden_output;
  diff = diff[0];
  // print average abs diff
  std::cout << "Average diff: " << diff.abs().mean().item<float>() << std::endl;

  return 0;
}