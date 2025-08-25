//
// Created by Entropy Xu on 9/9/24.
//

#include "prose_golden.h"
#include <torch_util.h>
#include "util.h"

#ifdef USE_TORCH

#include <torch/torch.h>

/**
 * @brief Perform a matrix multiplication operation on the input tensors with
 * exponentiation activation We perform the operation: out = exp(norm *
 * (activations * weights) + bias)
 * @param activations input tensor
 * @param weights weights tensor
 * @param out output tensor
 * @param bias per-row pre-activation bias tensor
 * @param norms per-row pre-activation normalization tensor
 * @param chosen_batch_size chosen batch size
 * @param M # of rows in the input tensor (activation)
 * @param K # of columns in the input tensor (activation) and # of rows in the
 * weights tensor
 * @param N # of columns in the weights tensor
 * @param write_out output tensor for the normalization values provided by
 * softmax
 */
void prose_isa_golden::prose_e_matmul(float* activations, float* weights,
                                      float* out, float* bias, float* norms,
                                      int chosen_batch_size, int M, int K,
                                      int N, float* write_out) {
  torch::Tensor activations_tensor =
      torch::from_blob(activations, {chosen_batch_size, M, K}, torch::kFloat32);
  torch::Tensor weights_tensor =
      torch::from_blob(weights, {K, N}, torch::kFloat32);
  torch::Tensor bias_tensor = torch::from_blob(bias, {N}, torch::kFloat32);
  torch::Tensor norms_tensor = torch::from_blob(norms, {M}, torch::kFloat32);

  // Perform matrix multiplication: activations * weights
  torch::Tensor result = torch::matmul(activations_tensor, weights_tensor);

  // Add bias to each row (broadcasted addition)
  result = result + bias_tensor;

  // Apply per-row normalization (broadcasted multiplication)
  result = result * norms_tensor.view({-1, 1});

  // Apply exponentiation activation
  result = torch::exp(result);

  // Write the result back to the output pointer
  std::memcpy(out, result.data_ptr<float>(), result.numel() * sizeof(float));

  // Optionally write out the normalization values (softmax, if requested)
  if (write_out) {
    torch::Tensor softmax_result = torch::softmax(result, -1);
    std::memcpy(write_out, softmax_result.data_ptr<float>(),
                softmax_result.numel() * sizeof(float));
  }
}

/**
 * @brief Perform a matrix multiplication operation on the input tensors with NO
 * activation We perform the operation: out = norm * (activations * weights) +
 * bias
 * @param activations
 * @param weights
 * @param out
 * @param bias
 * @param chosen_batch_size
 * @param M
 * @param K
 * @param N
 * @param output_transpose
 * @param norms
 */
void prose_isa_golden::prose_m_matmul(float* activations, float* weights,
                                      float* out, float* bias,
                                      int chosen_batch_size, int M, int K,
                                      int N, bool output_transpose,
                                      float* norms) {
  // Create Torch tensors from the remote pointers
  torch::Tensor activations_tensor =
      torch::from_blob(activations, {chosen_batch_size, M, K}, torch::kFloat32);
  torch::Tensor weights_tensor =
      torch::from_blob(weights, {K, N}, torch::kFloat32);
  // check if bias and norms are nullptr
  torch::Tensor bias_tensor = torch::zeros({N}, torch::kFloat32);
  if (bias != nullptr) {
    bias_tensor = torch::from_blob(bias, {N}, torch::kFloat32);
  }
  torch::Tensor norms_tensor = torch::ones({M}, torch::kFloat32);
  if (norms != nullptr) {
    norms_tensor = torch::from_blob(norms, {M}, torch::kFloat32);
  }

  // Perform matrix multiplication: activations * weights
  // activations_tensor shape: (batch_size, M, K)
  // weights_tensor shape: (K, N)
  // result shape: (batch_size, M, N)
  torch::Tensor result = torch::matmul(activations_tensor, weights_tensor);

  // Apply per-row normalization
  // norms_tensor shape: (M)
  // To broadcast norms across the N dimension, reshape it to (1, M, 1)
  result = result * norms_tensor.view({1, M, 1});

  // Add bias to each row
  // bias_tensor shape: (N)
  // To broadcast bias across the batch and M dimensions, reshape it to (1, 1,
  // N)
  result = result + bias_tensor.view({1, 1, N});

  if (output_transpose)
    std::cerr << "Output should be transposed, but this is not implemented yet."
              << std::endl;

  // Write the result back to the output pointer
  // Calculate the number of elements to copy
  size_t num_elements = result.numel();
  std::memcpy(out, result.data_ptr<float>(), num_elements * sizeof(float));
}

/**
 * @brief Perform a matrix multiplication operation on the input tensors with
 * GeLU activation We perform the operation: out = GeLU(norm * (activations *
 * weights) + bias)
 * @param activations input tensor
 * @param weights weights tensor
 * @param norm per-row pre-activation normalization tensor
 * @param bias per-row pre-activation bias tensor
 * @param out output tensor
 * @param chosen_batch_size chosen batch size
 * @param output_transpose whether to transpose the output. Importantly, if the
 * norm and bias are set to valid pointers, then the output should be
 * transposed. Otherwise, the bias and normalization will be applied by-column
 * instead of by-row. Ask me for clarification if this doesn't make sense.
 * @param M # of rows in the input tensor (activation)
 * @param K # of columns in the input tensor (activation) and # of rows in the
 * weights tensor
 * @param N # of columns in the weights tensor
 */
void prose_isa_golden::prose_g_matmul(float* activations, float* weights,
                                      float* norm, float* bias, float* out,
                                      int chosen_batch_size, int M, int K,
                                      int N) {
  // Step 1: Create Torch tensors from the remote pointers
  // Note: Ensure that the float* can be safely cast to float*
  torch::Tensor activations_tensor =
      torch::from_blob(activations, {chosen_batch_size, M, K}, torch::kFloat32);
  torch::Tensor weights_tensor =
      torch::from_blob(weights, {K, N}, torch::kFloat32);

  // Step 2: Perform matrix multiplication: activations * weights
  // activations_tensor shape: (batch_size, M, K)
  // weights_tensor shape: (K, N)
  // result shape: (batch_size, M, N)
  torch::Tensor result = torch::matmul(
      activations_tensor, weights_tensor); // Shape: (batch_size, M, N)

  // Step 3: Apply normalization if provided
  if (norm != nullptr) {
    // norm shape: (M)
    torch::Tensor norms_tensor = torch::from_blob(norm, {M}, torch::kFloat32);

    // Reshape norms_tensor to (1, M, 1) for broadcasting
    torch::Tensor norm_broadcast = norms_tensor.view({1, M, 1});

    // Apply normalization: result = result * norm
    result = result * norm_broadcast;
  }

  // Step 4: Apply bias if provided
  if (bias != nullptr) {
    // bias shape: (N)
    torch::Tensor bias_tensor = torch::from_blob(bias, {N}, torch::kFloat32);

    // Reshape bias_tensor to (1, 1, N) for broadcasting
    torch::Tensor bias_broadcast = bias_tensor.view({1, 1, N});

    // Apply bias: result = result + bias
    result = result + bias_broadcast;
  }

  // Step 5: Apply GeLU activation
  result = torch::gelu(result);

  // Step 6: Write the result back to the output pointer
  // Ensure that 'out' points to a memory region large enough to hold the result
  std::memcpy(out, result.data_ptr<float>(), result.numel() * sizeof(float));
}

/**
 * @brief Perform a self attention operation on the input tensor. This does not
 * use kv-caching, that needs to be done manually using the other operations.
 *
 * The input shape is (batch_size, input_length, D). The output shape is
 * (batch_size, input_length, input_length).
 *
 * @param input The input tensor.
 * @param batch_size The batch size of the input tensor.
 * @param input_length The number of features in the input tensor.
 * @param D The dimension of the tensor
 * @param query The query weights
 * @param key The key weights
 * @param value The value weights
 * @param query_bias The query bias
 * @param key_bias The key bias
 * @param value_bias The value bias
 * @param temp The temporary tensors: We need space to compute in * Kw, in * Vw,
 * K * V, etc... If we need more space for temporary tensor, we can modify this,
 * of course.
 * @param out The output tensor.
 */

static void print_a_bit(torch::Tensor a) {
  int l = a.sizes()[2];
  auto* p = a.data_ptr<float>();
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) {
      printf("%0.4f\t", p[i * l + j]);
    }
    printf("\n");
  }
}

void prose_isa_golden::prose_self_attention(
    float* input, uint8_t batch_size, uint16_t input_length, uint16_t D,
    uint16_t head_size, float* query, float* key, float* value, float* out_proj,
    float* query_bias, float* key_bias, float* value_bias, float* causal_mask,
    float* out_accumulation) {
  // Convert float* to torch::Tensor
  torch::Tensor input_tensor =
      torch::from_blob(input, {batch_size, input_length, D}, torch::kFloat32);
  torch::Tensor query_weights =
      torch::from_blob(query, {head_size, D}, torch::kFloat32);
  torch::Tensor key_weights =
      torch::from_blob(key, {head_size, D}, torch::kFloat32);

  torch::Tensor value_weights =
      torch::from_blob(value, {head_size, D}, torch::kFloat32);
  torch::Tensor query_bias_tensor =
      torch::from_blob(query_bias, {D}, torch::kFloat32);
  torch::Tensor key_bias_tensor =
      torch::from_blob(key_bias, {D}, torch::kFloat32);
  torch::Tensor value_bias_tensor =
      torch::from_blob(value_bias, {D}, torch::kFloat32);
  torch::Tensor out_proj_tensor =
      torch::from_blob(out_proj, {D, head_size}, torch::kFloat32);
  torch::Tensor causal_mask_tensor = torch::from_blob(
      causal_mask, {batch_size, input_length, input_length}, torch::kFloat32);
  torch::Tensor out_accumulation_tensor = torch::from_blob(
      out_accumulation, {batch_size, input_length, D}, torch::kFloat32);

  // Compute the query, key, value projections
  torch::Tensor query_tensor = query_bias == nullptr
      ? torch::matmul(input_tensor, query_weights.transpose(-1, -2))
      : torch::matmul(input_tensor, query_weights.transpose(-1, -2)) +
          query_bias_tensor;
  torch::Tensor key_tensor = key_bias == nullptr
      ? torch::matmul(input_tensor, key_weights.transpose(-1, -2))
      : torch::matmul(input_tensor, key_weights.transpose(-1, -2)) +
          key_bias_tensor;
  torch::Tensor value_tensor = value_bias == nullptr
      ? torch::matmul(input_tensor, value_weights.transpose(-1, -2))
      : torch::matmul(input_tensor, value_weights.transpose(-1, -2)) +
          value_bias_tensor;

  //  std::cout << "GOLDEN VALUE" << std::endl;
  //  print_batched_2d_tensor(value_tensor);

  // Calculate the attention scores (query * key^T)
  torch::Tensor attention_scores =
      torch::matmul(query_tensor, key_tensor.transpose(-1, -2));

  //  std::cout << "GOLDEN ATTENTION SCORES" << std::endl;
  // Apply the causal mask
  attention_scores += causal_mask_tensor;
  // Apply softmax to get the attention weight matrix
  torch::Tensor attention_weights = torch::softmax(attention_scores, -1);
  //  std::cout << "GOLDEN ATTENTION WEIGHTS" << std::endl;
  //  print_batched_2d_tensor(attention_weights);
  // Compute the attention output: attention_weights * value
  torch::Tensor attention_output =
      torch::matmul(attention_weights, value_tensor);
  //  std::cout << "GOLDEN ATTENTION OUT" << std::endl;
  //  print_batched_2d_tensor(attention_output);

  // std::cout << "ATTENTION TO OUTPUT:" << std::endl;
  // print_batched_2d_tensor(attention_output);
  // std::cout << "OUTPUT_PROJ_TENSOR:" << std::endl;
  // auto q = out_proj_tensor.transpose(-1, -2);
  // print_2d_tensor(q);
  // exit(0);

  // compute the output projection
  torch::Tensor final_output =
      torch::matmul(attention_output, out_proj_tensor.transpose(-1, -2));
  // std::cout << "OUTPUT:" << std::endl;
  // print_batched_2d_tensor(final_output);
  //  std::cout << "GOLDEN FINAL OUT" << std::endl;
  //  print_batched_2d_tensor(final_output);
  //  std::cout << "OUTPUT ACCUMULATION TENSOR BEFORE GOLDEN UPDATE : " <<
  //  out_proj_tensor.sizes() << std::endl;
  //  print_batched_2d_tensor(out_accumulation_tensor);
  // accumulate into
  std::memcpy(out_accumulation, out_accumulation_tensor.data_ptr<float>(),
              out_accumulation_tensor.numel() * sizeof(float));

  // accumulate into the output tensor
  out_accumulation_tensor = final_output + out_accumulation_tensor;
  // std::cout << "OUTPUT ACCUMULATION TENSOR AFTER GOLDEN UPDATE : " <<
  // out_proj_tensor.sizes() << std::endl;
  // print_batched_2d_tensor(out_accumulation_tensor);
  // accumulate into
  std::memcpy(out_accumulation, out_accumulation_tensor.data_ptr<float>(),
              out_accumulation_tensor.numel() * sizeof(float));
}
static void print_a_bit(float* f) {
  int l = 768;
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) {
      printf("%0.4f\t", f[i * l + j]);
    }
    printf("\n");
  }
}


/**
 * @brief Perform a layer normalization operation on the input tensor.
 * Arguments correspond to
 * https://pytorch.org/docs/stable/generated/torch.nn.LayerNorm.html
 * @param input The input tensor.
 * @param gamma
 * @param beta
 * @param batch_size
 * @param input_length
 * @param out normalized output tensor
 */
void prose_isa_golden::prose_layer_norm(float* input, float* gamma, float* beta,
                                        uint8_t batch_size,
                                        uint16_t input_length, uint16_t D,
                                        float* out) {
  // Convert float* to torch::Tensor
  torch::Tensor input_tensor =
      torch::from_blob(input, {batch_size, input_length, D}, torch::kFloat32);
  torch::Tensor gamma_tensor = torch::from_blob(gamma, {D}, torch::kFloat32);
  torch::Tensor beta_tensor = torch::from_blob(beta, {D}, torch::kFloat32);

  // Compute layer normalization
  torch::Tensor normalized =
      torch::layer_norm(input_tensor, {D}, gamma_tensor, beta_tensor, 1e-5);

  // Copy the result back to the output pointer
  std::memcpy(out, normalized.data_ptr<float>(),
              normalized.numel() * sizeof(float));
}

void prose_isa_golden::prose_multi_head_attention(
    float* input, float* q_proj_w, float* k_proj_w, float* v_proj_w,
    float* o_proj_w, float* o_proj_b, float* causal_mask, uint8_t batch_size,
    uint16_t seq_len, uint16_t embed_dim, uint16_t num_heads,
    float* attn_output) {
  auto head_size = embed_dim / num_heads;
  assert(embed_dim == head_size * num_heads);
  // ensure that the attn_output is zero-initialized
  for (int i = 0; i < batch_size * seq_len * embed_dim; i++) {
    if (attn_output[i] != 0) {
      throw std::runtime_error("attn_output is not zero-initialized");
    }
  }

  // Convert float* to torch::Tensor
  torch::Tensor input_tensor = torch::from_blob(
      input, {batch_size, seq_len, embed_dim}, torch::kFloat32);
  torch::Tensor q_proj_w_tensor =
      torch::from_blob(q_proj_w, {embed_dim, embed_dim}, torch::kFloat32);
  torch::Tensor k_proj_w_tensor =
      torch::from_blob(k_proj_w, {embed_dim, embed_dim}, torch::kFloat32);
  torch::Tensor v_proj_w_tensor =
      torch::from_blob(v_proj_w, {embed_dim, embed_dim}, torch::kFloat32);
  torch::Tensor o_proj_w_tensor =
      torch::from_blob(o_proj_w, {embed_dim, embed_dim}, torch::kFloat32);
  torch::Tensor o_proj_b_tensor =
      torch::from_blob(o_proj_b, {embed_dim}, torch::kFloat32);
  torch::Tensor causal_mask_tensor =
      torch::from_blob(causal_mask, {1, seq_len, seq_len}, torch::kFloat32);
  torch::Tensor attn_output_tensor = torch::from_blob(
      attn_output, {batch_size, seq_len, embed_dim}, torch::kFloat32);

  // preload o_proj_b into attn_output
  attn_output_tensor += o_proj_b_tensor;

  for (int h = 0; h < num_heads; h++) {
    auto q_proj_w_h =
        q_proj_w_tensor.slice(0, h * head_size, (h + 1) * head_size)
            .contiguous();
    auto k_proj_w_h =
        k_proj_w_tensor.slice(0, h * head_size, (h + 1) * head_size)
            .contiguous();
    auto v_proj_w_h =
        v_proj_w_tensor.slice(0, h * head_size, (h + 1) * head_size)
            .contiguous();
    auto o_proj_w_h =
        o_proj_w_tensor.slice(1, h * head_size, (h + 1) * head_size)
            .contiguous();

    prose_isa_golden::prose_self_attention(
        input_tensor.data_ptr<float>(), batch_size, seq_len, embed_dim,
        head_size, q_proj_w_h.data_ptr<float>(), k_proj_w_h.data_ptr<float>(),
        v_proj_w_h.data_ptr<float>(), o_proj_w_h.data_ptr<float>(), nullptr,
        nullptr, nullptr, causal_mask_tensor.data_ptr<float>(),
        attn_output_tensor.data_ptr<float>());

    std::cout << "GOLDEN h" << h << std::endl;
    print_batched_2d_tensor(attn_output_tensor);
    // if (h == 1) exit(0);
  }
  // Copy the result back to the output pointer
  std::memcpy(attn_output, attn_output_tensor.data_ptr<float>(),
              attn_output_tensor.numel() * sizeof(float));
}

void prose_isa_golden::prose_matadd(float* a, float* b, float* c, int sz) {
  for (int i = 0; i < sz; ++i)
    c[i] = a[i] + b[i];
}

const int m_mlp = 3072;

void prose_isa_golden::prose_mlp(float* input, float* fc_wgt, float* fc_bias,
                                 float* proj_wgt, float* proj_bias, float* out,
                                 int batch_size, int seq_len, int embed_size) {
  auto act = torch::from_blob(input, {batch_size, seq_len, embed_size});
  auto t_fc = torch::from_blob(fc_wgt, {embed_size, m_mlp});
  auto t_fc_b = torch::from_blob(fc_bias, {m_mlp});
  auto t_pw = torch::from_blob(proj_wgt, {m_mlp, embed_size});
  auto t_pb = torch::from_blob(proj_bias, {embed_size});

  auto prod1 = torch::matmul(act, t_fc) + t_fc_b;
  prod1 = torch::gelu(prod1);
  auto res = torch::matmul(prod1, t_pw) + t_pb;
  memcpy(out, res.contiguous().data_ptr<float>(),
         sizeof(float) * seq_len * embed_size * batch_size);
}

void prose_isa_golden::decoder_layer(
    float* input, float* ln1_wgt, float* ln1_bias, float* ln2_wgt,
    float* ln2_bias, float* qarray, float* karray, float* varray, float* oarray,
    float* obias, float* causal_mask, float* mlp_fc_wgt, float* mlp_fc_bias,
    float* mlp_proj_wgt, float* mlp_proj_bias, uint8_t batch_size,
    uint16_t seq_len, uint16_t embed_dim, uint16_t num_heads,
    float* attn_output) {
  float* interm = new float[batch_size * seq_len * embed_dim];
  float* interm2 = new float[batch_size * seq_len * embed_dim];
  float* residual = new float[batch_size * seq_len * embed_dim];

  memset(interm2, 0, sizeof(float) * batch_size * seq_len * embed_dim);
  prose_isa_golden::prose_layer_norm(input, ln1_wgt, ln1_bias, batch_size,
                                     seq_len, embed_dim, interm);

  // attn
  prose_isa_golden::prose_multi_head_attention(
      interm, qarray, karray, varray, oarray, obias, causal_mask, batch_size,
      seq_len, embed_dim, num_heads, interm2);

  std::cout << "AFTER ATTENTION: " << std::endl;
  for (int i = 0; i < seq_len; ++i) {
    for (int j = 0; j < embed_dim; ++j) {
      float q = interm2[i * embed_dim + j];
      printf("%0.4f ", q);
    }
    printf("\n");
  }

  // add inputs to the attn output, call this residual
  prose_isa_golden::prose_matadd(interm2, input, residual,
                                 batch_size * seq_len * embed_dim);

  std::cout << "AFTER ADD RESIDUAL: " << std::endl;
  for (int i = 0; i < seq_len; ++i) {
    for (int j = 0; j < embed_dim; ++j) {
      float q = residual[i * embed_dim + j];
      printf("%0.4f ", q);
    }
    printf("\n");
  }

  prose_isa_golden::prose_layer_norm(residual, ln2_wgt, ln2_bias, batch_size,
                                     seq_len, embed_dim, interm2);

  std::cout << "AFTER LN2: " << std::endl;
  for (int i = 0; i < seq_len; ++i) {
    for (int j = 0; j < embed_dim; ++j) {
      float q = interm2[i * embed_dim + j];
      printf("%0.4f ", q);
    }
    printf("\n");
  }

  // THESE QUALIFIER ARE NOT REDUNDANT
  prose_isa_golden::prose_mlp(interm2, mlp_fc_wgt, mlp_fc_bias, mlp_proj_wgt,
                              mlp_proj_bias, interm, batch_size, seq_len,
                              embed_dim);
  std::cout << "AFTER MLP: " << std::endl;
  for (int i = 0; i < seq_len; ++i) {
    for (int j = 0; j < embed_dim; ++j) {
      float q = interm[i * embed_dim + j];
      printf("%0.4f ", q);
    }
    printf("\n");
  }

  prose_isa_golden::prose_matadd(residual, interm, attn_output, embed_dim * batch_size * seq_len);

  std::cout << "AFTER final residual add: " << std::endl;
  for (int i = 0; i < seq_len; ++i) {
    for (int j = 0; j < embed_dim; ++j) {
      float q = attn_output[i * embed_dim + j];
      printf("%0.4f ", q);
    }
    printf("\n");
  }

}

#endif
