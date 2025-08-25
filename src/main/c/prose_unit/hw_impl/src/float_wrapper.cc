#include "float_wrapper.h"
#include <bit>
#include <cmath>
#include <cstring>
#include "beethoven/rocc_cmd.h"
#include "beethoven_hardware.h"
#include "prose_impl.h"
#include "util.h"

/**
 * Chris Kjellqvist 10/6/24
 */
using namespace beethoven;

fpga_handle_t handle;


static remote_ptr allocBiasFromFloatPtr(float* ptr, int M, int N,
                                        int batch_size, int bias_mode) {
  int eles;
  int batch;

  if (bias_mode == PROSE_biasMATRIX) {
    eles = M * N;
    batch = 1;
  } else if (bias_mode == PROSE_biasCOLS) {
    eles = N;
    batch = 1;
  } else if (bias_mode == PROSE_biasBATCHEDMATRIX) {
    eles = M * N * batch_size;
    batch = batch_size;
  } else {
    std::cout << "Tried to alloc nothing..." << std::endl;
    exit(1);
  }
  auto bias_ptr = handle.malloc(eles * 2);
  uint16_t* tmp = new uint16_t[eles];
  memcpy_fp32_to_bf16(tmp, ptr, eles);
  if (bias_mode != PROSE_biasCOLS) {
    convertRowMajorFormatToProSEColMajor(tmp, (uint16_t*)bias_ptr.getHostAddr(),
                                         M, N, batch, PROSE_Nmin);
  } else {
    memcpy(bias_ptr.getHostAddr(), tmp, sizeof(uint16_t) * eles);
  }

  delete[] tmp;

  handle.copy_to_fpga(bias_ptr);

  return bias_ptr;
}

void prose_float_wrapper::prose_e_matmul(
    std::variant<uint16_t*, float*, beethoven::remote_ptr> activations,
    std::variant<uint16_t*, float*, beethoven::remote_ptr> weights,
    std::variant<uint16_t*, float*, beethoven::remote_ptr> out,
    std::variant<float*, beethoven::remote_ptr>
        bias, // instead of a bias vector, this is now a matrix
    float* norms, bool weights_are_batched, int chosen_batch_size, int M, int K,
    int N, std::variant<float*, beethoven::remote_ptr> write_out, int biasMode,
    bool norm_per_batch) {
  remote_ptr in_ptr, weights_ptr, out_ptr, norm_out;
  if (std::holds_alternative<float*>(write_out)) {
    norm_out = handle.malloc(2 * M * chosen_batch_size);
  } else {
    norm_out = std::get<remote_ptr>(write_out);
  }

  auto temp_ptr = new uint16_t[3072 * 768];
  remote_ptr* norm_ptr = nullptr;
  if (norms != nullptr) {
    norm_ptr = new remote_ptr;
    *norm_ptr = handle.malloc(2 * M * (norm_per_batch ? chosen_batch_size : 1));
    memcpy_fp32_to_bf16((uint16_t*)norm_ptr->getHostAddr(), norms, M);
    handle.copy_to_fpga(*norm_ptr);
  }
  remote_ptr bias_ptr;
  bool use_bias_matrix = biasMode != PROSE_biasNONE;
  if (use_bias_matrix) {
    if (std::holds_alternative<float*>(bias)) {
      float* ptr = std::get<float*>(bias);
      bias_ptr = allocBiasFromFloatPtr(ptr, M, N, chosen_batch_size, biasMode);
    } else {
      bias_ptr = std::get<remote_ptr>(bias);
    }
  }

  if (std::holds_alternative<float*>(activations)) {
    in_ptr = handle.malloc(2 * M * K * chosen_batch_size);
    // input is a float* but we need to convert it to a uint16_t* for the FPGA
    memcpy_fp32_to_bf16(temp_ptr, std::get<float*>(activations),
                        M * K * chosen_batch_size);
    convertRowMajorFormatToProSEColMajor(temp_ptr,
                                         (uint16_t*)in_ptr.getHostAddr(), M, K,
                                         chosen_batch_size, PROSE_Nmin);
    handle.copy_to_fpga(in_ptr);
  } else if (std::holds_alternative<uint16_t*>(activations)) {
    // input is already in the right format
    in_ptr = handle.malloc(sizeof(uint16_t) * chosen_batch_size * M * K);
    memcpy(in_ptr.getHostAddr(), std::get<uint16_t*>(activations),
           M * K * chosen_batch_size * sizeof(uint16_t));
  } else {
    in_ptr = std::get<remote_ptr>(activations);
  }

  auto weight_size_eles = K * N * (weights_are_batched ? chosen_batch_size : 1);
  if (std::holds_alternative<float*>(weights)) {
    weights_ptr = handle.malloc(2 * weight_size_eles);
    memcpy_fp32_to_bf16(temp_ptr, std::get<float*>(weights), weight_size_eles);
    convertRowMajorFormatToProSERowMajor(
        temp_ptr, (uint16_t*)weights_ptr.getHostAddr(), K, N,
        (weights_are_batched ? chosen_batch_size : 1), PROSE_Nmin);
    // don't need to convert format because it's just a vector
  } else if (std::holds_alternative<uint16_t*>(weights)) {
    // weights are already in the right format
    weights_ptr = handle.malloc(2 * weight_size_eles);
    memcpy(weights_ptr.getHostAddr(), std::get<uint16_t*>(weights),
           weight_size_eles * sizeof(uint16_t));
  } else {
    weights_ptr = std::get<remote_ptr>(weights);
  }

  if (std::holds_alternative<float*>(out) ||
      std::holds_alternative<uint16_t*>(out)) {
    out_ptr = handle.malloc(2 * M * N * chosen_batch_size);
  } else {
    out_ptr = std::get<remote_ptr>(out);
  }
  // need this if we test on FPGA, UB otherwise (although it would be fine on
  // Kria without copying back and forward)
  handle.copy_to_fpga(weights_ptr);

  printf("E-TYPE:\n"
         "Pointers:\n"
         "\tinput: %lx\n"
         "\tweight: %lx\n"
         "\toutput: %lx\n"
         "\tbias: %lx\n"
         "\tnorm: %lx\n"
         "\tM, K, N: %d, %d %d\n",
         in_ptr.getFpgaAddr(), weights_ptr.getFpgaAddr(), out_ptr.getFpgaAddr(),
         use_bias_matrix ? bias_ptr.getFpgaAddr() : 0L,
         norms != nullptr ? norm_ptr->getFpgaAddr() : 0L, M, K, N);

  prose_impl::prose_e_matmul(in_ptr, weights_ptr, out_ptr, &bias_ptr, norm_ptr,
                             biasMode, chosen_batch_size, weights_are_batched,
                             M, K, N, norm_out, norm_per_batch);

  handle.copy_from_fpga(out_ptr);
  delete[] temp_ptr;
  if (norms != nullptr) {
    delete norm_ptr;
  }

  if (std::holds_alternative<float*>(out)) {
    // convert from uint16_t & special memory format to normal array
    convertPCMtoTCM((uint16_t*)out_ptr.getHostAddr(), std::get<float*>(out), M,
                    N, chosen_batch_size, PROSE_Nmin);
  } else if (std::holds_alternative<uint16_t*>(out)) {
    memcpy(std::get<uint16_t*>(out), out_ptr.getHostAddr(),
           M * N * chosen_batch_size * sizeof(float));
  } else {
    // do nothing because the remote pointer allocation already holds the data
  }
  if (std::holds_alternative<float*>(write_out)) {
    // convert from uint16_t & special memory format to normal array
    float* wo = std::get<float*>(write_out);
    for (int i = 0; i < M * chosen_batch_size; ++i) {
      wo[i] = bf16_to_float(((uint16_t*)norm_out.getHostAddr())[i]);
    }
  } else {
    // do nothing because the remote pointer allocation already holds the data
  }
}

void prose_float_wrapper::prose_m_matmul(
    std::variant<uint16_t*, float*, beethoven::remote_ptr> activations,
    std::variant<uint16_t*, float*, beethoven::remote_ptr> weights,
    std::variant<uint16_t*, float*, beethoven::remote_ptr> out,
    std::variant<float*, beethoven::remote_ptr> bias,
    std::variant<float*, beethoven::remote_ptr> norms, int chosen_batch_size,
    int M, int K, int N, bool output_transpose, bool norm_per_batch,
    bool weights_are_batched, int biasMode) {
  remote_ptr in_ptr, weights_ptr, out_ptr;
  auto temp_ptr = new uint16_t[3072 * 768 * chosen_batch_size];
  remote_ptr* norm_ptr = nullptr;
  if (std::holds_alternative<float*>(norms)) {
    auto norm = std::get<float*>(norms);
    if (norm != nullptr) {
      norm_ptr = new remote_ptr;
      *norm_ptr = handle.malloc(2 * M);
      memcpy_fp32_to_bf16((uint16_t*)norm_ptr->getHostAddr(), norm, M);
      handle.copy_to_fpga(*norm_ptr);
    }
  } else {
    norm_ptr = new remote_ptr;
    *norm_ptr = std::get<remote_ptr>(norms);
  }
  remote_ptr bias_ptr;
  bool use_bias_matrix = biasMode != PROSE_biasNONE;
  if (use_bias_matrix) {
    if (std::holds_alternative<float*>(bias)) {
      float* ptr = std::get<float*>(bias);
      bias_ptr = allocBiasFromFloatPtr(ptr, M, N, chosen_batch_size, biasMode);
    } else {
      bias_ptr = std::get<remote_ptr>(bias);
    }
  }
  /** MOVE ACTIVATIONS **/
  if (std::holds_alternative<float*>(activations)) {
    in_ptr = handle.malloc(2 * M * K * chosen_batch_size);
    memcpy_fp32_to_bf16(temp_ptr, std::get<float*>(activations),
                        M * K * chosen_batch_size);
    convertRowMajorFormatToProSEColMajor(temp_ptr,
                                         (uint16_t*)in_ptr.getHostAddr(), M, K,
                                         chosen_batch_size, PROSE_Nmin);
    handle.copy_to_fpga(in_ptr);
  } else if (std::holds_alternative<uint16_t*>(activations)) {
    in_ptr = handle.malloc(2 * M * K * chosen_batch_size);
    memcpy(in_ptr.getHostAddr(), std::get<uint16_t*>(activations),
           M * K * chosen_batch_size * sizeof(uint16_t));
  } else {
    in_ptr = std::get<remote_ptr>(activations);
  }
  /** MOVE WEIGHTS **/
  auto weights_eles = K * N * (weights_are_batched ? chosen_batch_size : 1);
  if (std::holds_alternative<float*>(weights)) {
    weights_ptr = handle.malloc(2 * weights_eles);
    memcpy_fp32_to_bf16(temp_ptr, std::get<float*>(weights), weights_eles);
    convertRowMajorFormatToProSERowMajor(
        temp_ptr, (uint16_t*)weights_ptr.getHostAddr(), K, N,
        (weights_are_batched ? chosen_batch_size : 1), PROSE_Nmin);
    // don't need to convert format because it's just a vector
  } else if (std::holds_alternative<uint16_t*>(weights)) {
    weights_ptr = handle.malloc(2 * weights_eles);
    memcpy(weights_ptr.getHostAddr(), std::get<uint16_t*>(weights),
           weights_eles * sizeof(uint16_t));
  } else {
    weights_ptr = std::get<remote_ptr>(weights);
  }
  // need this if we test on FPGA, UB otherwise (although it would be fine on
  // Kria without copying back and forward)
  handle.copy_to_fpga(weights_ptr);

  if (std::holds_alternative<float*>(out) ||
      std::holds_alternative<uint16_t*>(out)) {
    out_ptr = handle.malloc(2 * M * N * chosen_batch_size);
  } else {
    out_ptr = std::get<remote_ptr>(out);
  }

  printf("M-TYPE:\n"
         "Pointers:\n"
         "\tBatch Size: %d\n"
         "\tinput: %lx\n"
         "\tweight: %lx\n"
         "\toutput: %lx\n"
         "\tbias: %lx\n"
         "\tnorm: %lx\n"
         "\tM, K, N: %d, %d %d\n",
         chosen_batch_size, in_ptr.getFpgaAddr(), weights_ptr.getFpgaAddr(),
         out_ptr.getFpgaAddr(), use_bias_matrix ? bias_ptr.getFpgaAddr() : 0L,
         norm_ptr != nullptr ? norm_ptr->getFpgaAddr() : 0L, M, K, N);

  prose_impl::prose_m_matmul(in_ptr, weights_ptr, out_ptr, &bias_ptr, biasMode,
                             chosen_batch_size, M, K, N, output_transpose,
                             norm_ptr, weights_are_batched, norm_per_batch);
  handle.copy_from_fpga(out_ptr);
  delete[] temp_ptr;
  if (std::holds_alternative<float*>(norms)) {
    delete norm_ptr;
  }

  if (std::holds_alternative<float*>(out)) {
    // convert from uint16_t & special memory format to normal array
    convertPCMtoTCM((uint16_t*)out_ptr.getHostAddr(), std::get<float*>(out), M,
                    N, chosen_batch_size, PROSE_Nmin);
  } else if (std::holds_alternative<uint16_t*>(out)) {
    memcpy(std::get<uint16_t*>(out), out_ptr.getHostAddr(),
           M * N * chosen_batch_size * sizeof(float));
  } else {
    // do nothing because the remote pointer allocation already holds the data
  }
}

void prose_float_wrapper::prose_g_matmul(
    std::variant<uint16_t*, float*, beethoven::remote_ptr> activations,
    std::variant<uint16_t*, float*, beethoven::remote_ptr> weights,
    std::variant<uint16_t*, float*, beethoven::remote_ptr> out,
    std::variant<float*, beethoven::remote_ptr> bias, float* norms,
    int biasMode, int chosen_batch_size, int M, int K, int N,
    bool per_batch_norm) {
  //  auto in_ptr = handle.malloc(2 * M * K);
  //  auto weights_ptr = handle.malloc(2 * K * N);
  //  auto out_ptr = handle.malloc(2 * M * N);
  remote_ptr in_ptr, weights_ptr, out_ptr;
  auto temp_ptr = new uint16_t[3072 * 768];
  remote_ptr* norm_ptr = nullptr;
  if (norms != nullptr) {
    norm_ptr = new remote_ptr;
    *norm_ptr = handle.malloc(2 * M);
    memcpy_fp32_to_bf16((uint16_t*)norm_ptr->getHostAddr(), norms, M);
    handle.copy_to_fpga(*norm_ptr);
  }
  remote_ptr bias_ptr;
  bool use_bias_matrix = biasMode != PROSE_biasNONE;
  if (use_bias_matrix) {
    if (std::holds_alternative<float*>(bias)) {
      float* ptr = std::get<float*>(bias);
      bias_ptr = allocBiasFromFloatPtr(ptr, M, N, chosen_batch_size, biasMode);
    } else {
      bias_ptr = std::get<remote_ptr>(bias);
    }
  }

  /** MOVE ACTIVATIONS **/
  if (std::holds_alternative<float*>(activations)) {
    in_ptr = handle.malloc(2 * M * K * chosen_batch_size);
    memcpy_fp32_to_bf16(temp_ptr, std::get<float*>(activations),
                        M * K * chosen_batch_size);
    convertRowMajorFormatToProSEColMajor(temp_ptr,
                                         (uint16_t*)in_ptr.getHostAddr(), M, K,
                                         chosen_batch_size, PROSE_Nmin);
    handle.copy_to_fpga(in_ptr);
  } else if (std::holds_alternative<uint16_t*>(activations)) {
    in_ptr = handle.malloc(2 * M * K * chosen_batch_size);
    memcpy(in_ptr.getHostAddr(), std::get<uint16_t*>(activations),
           M * K * chosen_batch_size * sizeof(uint16_t));
  } else {
    in_ptr = std::get<remote_ptr>(activations);
  }
  /** MOVE WEIGHTS **/
  if (std::holds_alternative<float*>(weights)) {
    weights_ptr = handle.malloc(2 * K * N);
    memcpy_fp32_to_bf16(temp_ptr, std::get<float*>(weights), K * N);
    convertRowMajorFormatToProSERowMajor(
        temp_ptr, (uint16_t*)weights_ptr.getHostAddr(), K, N, 1, PROSE_Nmin);
    // don't need to convert format because it's just a vector
  } else if (std::holds_alternative<uint16_t*>(weights)) {
    weights_ptr = handle.malloc(2 * K * N);
    memcpy(weights_ptr.getHostAddr(), std::get<uint16_t*>(weights),
           K * N * sizeof(uint16_t));
  } else {
    weights_ptr = std::get<remote_ptr>(weights);
  }
  // need this if we test on FPGA, UB otherwise (although it would be fine on
  // Kria without copying back and forward)
  handle.copy_to_fpga(weights_ptr);

  if (std::holds_alternative<float*>(out) ||
      std::holds_alternative<uint16_t*>(out)) {
    out_ptr = handle.malloc(2 * M * N * chosen_batch_size);
  } else {
    out_ptr = std::get<remote_ptr>(out);
  }

  printf("G-TYPE:\n"
         "Pointers:\n"
         "\tinput: %lx\n"
         "\tweight: %lx\n"
         "\toutput: %lx\n"
         "\tbias: %lx\n"
         "\tnorm: %lx\n"
         "\tM, K, N: %d, %d %d\n",
         in_ptr.getFpgaAddr(), weights_ptr.getFpgaAddr(), out_ptr.getFpgaAddr(),
         use_bias_matrix ? bias_ptr.getFpgaAddr() : 0L,
         norms != nullptr ? norm_ptr->getFpgaAddr() : 0L, M, K, N);

  prose_impl::prose_g_matmul(in_ptr, weights_ptr, norm_ptr, &bias_ptr, out_ptr,
                             chosen_batch_size, M, K, N, per_batch_norm,
                             biasMode);
  handle.copy_from_fpga(out_ptr);

  delete[] temp_ptr;
  if (norms != nullptr) {
    delete norm_ptr;
  }

  if (std::holds_alternative<float*>(out)) {
    // convert from uint16_t & special memory format to normal array
    convertPCMtoTCM((uint16_t*)out_ptr.getHostAddr(), std::get<float*>(out), M,
                    N, chosen_batch_size, PROSE_Nmin);
  } else if (std::holds_alternative<uint16_t*>(out)) {
    memcpy(std::get<uint16_t*>(out), out_ptr.getHostAddr(),
           M * N * chosen_batch_size * sizeof(float));
  } else {
    // do nothing because the remote pointer allocation already holds the data
  }
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
void prose_float_wrapper::prose_self_attention(
    float* input, uint8_t batch_size, uint16_t input_length, uint16_t D,
    uint16_t head_size, float* query, float* key, float* value,
    float* query_bias, float* key_bias, float* value_bias,
    float* attention_mask, float* sqrt_vector, float* output_bias,
    float* output_proj_tensor, bool use_output_bias,
    remote_ptr out_accumulation) {
  remote_ptr temps[4];
  for (int i = 0; i < 3; ++i) {
    temps[i] = handle.malloc(2 * head_size * input_length * batch_size);
  }
  temps[3] = handle.malloc(2 * D);

  // QUERY PROJECTION
  prose_m_matmul(input, query, temps[0], query_bias, nullptr, batch_size,
                 input_length, D, head_size, true, false, false,
                 (query_bias == nullptr) ? PROSE_biasNONE : PROSE_biasCOLS);
  // NOTE: previously, I thought that the operation was going to be (A x B) x (C
  // x D) -> this requires outputing the intermediate result in a different
  // format that the input (col major vs row major) From a memory layout
  // perspective, it corresponds to a matrix transpose. BUT, since we're really
  // doing (A x B) x (C x D)^T, this means that from the memory layout
  // perspective, we are storing (C x D)^T^T, which is the same as C x D in the
  // default output format KEY PROJECTION
  prose_m_matmul(input, key, temps[1], key_bias, nullptr, batch_size,
                 input_length, D, head_size, true, false, false,
                 (key_bias == nullptr) ? PROSE_biasNONE : PROSE_biasCOLS);
  {
    // not leaked, these are smart ptrs
    remote_ptr attention_score_matrix_temp =
        handle.malloc(2 * batch_size * input_length * input_length);
    // SOFTMAX(QUERY X KEY^T)
    prose_e_matmul(temps[0], temps[1], attention_score_matrix_temp,
                   attention_mask, sqrt_vector,
                   true, // weights ARE BATCHED (input (key projection) x
                   // weights (query projection))
                   batch_size, input_length, head_size, input_length, temps[3],
                   PROSE_biasMATRIX, false);

    // VALUE PROJECTION
    prose_m_matmul(input, value, temps[2], value_bias, nullptr, batch_size,
                   input_length, D, head_size,
                   false, // transpose (yes) - same reason as previous NOTE
                   true, // norm-per-batch (no)
                   false, // weights are batched (no)
                   (value_bias == nullptr) ? PROSE_biasNONE : PROSE_biasCOLS);


    // ATTENTION OUTPUT
    prose_m_matmul(attention_score_matrix_temp, temps[2], temps[1], nullptr,
                   temps[3], batch_size, input_length, input_length, head_size,
                   true,
                   true, // normalize per batch with softmax norm factors
                   true, // weights are batched
                   PROSE_biasNONE);
  }


  //  auto temp_  = new float[batch_size * input_length * head_size];
  //  convertPCMtoTCM((uint16_t *) temps[1].getHostAddr(), temp_, input_length,
  //  head_size, batch_size, PROSE_Nmin); auto temp_tensor =
  //  torch::from_blob(temp_, {batch_size, input_length, head_size}); std::cout
  //  << "temp_tensor sizes: " << temp_tensor.sizes() << std::endl;
  //  print_batched_2d_tensor(temp_tensor);
  //  auto out_accumulation_host = new float[batch_size * input_length * D];
  //  std::cout << "PROSE BEFORE UPDATE: " << std::endl;
  //  handle.copy_from_fpga(out_accumulation);
  //  convertPCMtoTCM((uint16_t *) out_accumulation.getHostAddr(),
  //  out_accumulation_host, input_length, D, batch_size, PROSE_Nmin); auto
  //  out_accumulation_tensor = torch::from_blob(out_accumulation_host,
  //  {batch_size, input_length, D});
  //  print_batched_2d_tensor(out_accumulation_tensor);


  // on the first go-through we initialize the output accumulator with the
  // matmul + bias, then we accumulate on it
  if (use_output_bias) {
    prose_m_matmul(temps[1], output_proj_tensor, out_accumulation, output_bias,
                   nullptr, batch_size, input_length, head_size, D, true, false,
                   false, PROSE_biasCOLS);
  } else {
    prose_m_matmul(temps[1], output_proj_tensor, out_accumulation,
                   out_accumulation, nullptr, batch_size, input_length,
                   head_size, D, true, false, false, PROSE_biasBATCHEDMATRIX);
  }
  //
  //  std::cout << "PROSE AFTER UPDATE: " << std::endl;
  //  handle.copy_from_fpga(out_accumulation);
  //  convertPCMtoTCM((uint16_t *) out_accumulation.getHostAddr(),
  //  out_accumulation_host, input_length, D, batch_size, PROSE_Nmin);
  //  out_accumulation_tensor = torch::from_blob(out_accumulation_host,
  //  {batch_size, input_length, D});
  //  print_batched_2d_tensor(out_accumulation_tensor);
}

void prose_float_wrapper::prose_layer_norm(float* input, float* gamma,
                                           float* beta, uint8_t batch_size,
                                           uint16_t seq_len, uint16_t embed_dim,
                                           float* out) {
  auto gamma_beta_combined = handle.malloc(2 * 2 * embed_dim);
  // combine gamma and beta from decorder_ln_1_[weight/bias] in alternating
  // fashion
  for (int i = 0; i < embed_dim; i++) {
    ((uint16_t*)(gamma_beta_combined.getHostAddr()))[2 * i + 1] =
        (reinterpret_cast<uint32_t&>(gamma[i])) >> 16;
    ((uint16_t*)(gamma_beta_combined.getHostAddr()))[2 * i] =
        (reinterpret_cast<uint32_t&>(beta[i])) >> 16;
  }
  handle.copy_to_fpga(gamma_beta_combined);

  auto input_ptr = handle.malloc(2 * embed_dim * seq_len * batch_size);
  auto out_ptr = handle.malloc(2 * embed_dim * seq_len * batch_size);
  uint16_t* temp = new uint16_t[embed_dim * seq_len * batch_size];
  memcpy_fp32_to_bf16(temp, input, embed_dim * seq_len * batch_size);
  convertRowMajorFormatToProSEColMajor(temp, (uint16_t*)input_ptr.getHostAddr(),
                                       seq_len, embed_dim, batch_size,
                                       PROSE_Nmin);
  handle.copy_to_fpga(input_ptr);

  prose_impl::prose_layer_norm(input_ptr, gamma_beta_combined, batch_size,
                               embed_dim, seq_len, out_ptr);

  handle.copy_from_fpga(out_ptr);
  for (int i = 0; i < 32; ++i) {
    printf("%d %x\n", i, ((uint16_t*)out_ptr.getHostAddr())[i]);
  }
  convertPCMtoTCM((uint16_t*)out_ptr.getHostAddr(), out, seq_len, embed_dim,
                  batch_size, PROSE_Nmin);
  delete[] temp;
}

void prose_float_wrapper::prose_multi_head_attention(
    float* input, float** q_proj_w, float** k_proj_w, float** v_proj_w,
    float** o_proj_w, float* o_proj_b, float* causal_mask, uint8_t batch_size,
    uint16_t seq_len, uint16_t embed_dim, uint16_t num_heads,
    float* attn_output, bool scaled_attention) {
  auto head_size = embed_dim / num_heads;
  assert(embed_dim == head_size * num_heads);
  std::cout << std::dec << embed_dim << " / " << num_heads
            << " = head_size: " << head_size << std::endl;
  for (int i = 0; i < batch_size * seq_len * embed_dim; i++) {
    if (attn_output[i] != 0) {
      std::cerr << "attn_output is not zero-initialized" << std::endl;
      exit(1);
    }
  }

  float* sqrt_vector = new float[seq_len];
  for (int i = 0; i < seq_len; i++) {
    if (scaled_attention)
      sqrt_vector[i] = 1.0 / std::sqrt(seq_len);
    else
      sqrt_vector[i] = 1;
  }
  auto attn_accumulator = handle.malloc(2 * batch_size * seq_len * embed_dim);
  memset(attn_accumulator.getHostAddr(), 0,
         2 * batch_size * seq_len * embed_dim);

  for (int h = 0; h < num_heads; ++h) {
    printf("starting head %d\n", h);
    auto q_proj_w_h = q_proj_w[h];
    auto k_proj_w_h = k_proj_w[h];
    auto v_proj_w_h = v_proj_w[h];
    auto o_proj_w_h = o_proj_w[h];

    prose_self_attention(input, batch_size, seq_len, embed_dim, head_size,
                         q_proj_w_h, k_proj_w_h, v_proj_w_h, nullptr, nullptr,
                         nullptr, causal_mask, sqrt_vector, o_proj_b,
                         o_proj_w_h, h == 0, attn_accumulator);
    uint16_t * float_check = new uint16_t [seq_len * batch_size * embed_dim];
    convertPCMtoTCM((uint16_t*)attn_accumulator.getHostAddr(), float_check,
                seq_len, embed_dim, batch_size, PROSE_Nmin);
    for (int i = 0; i < seq_len; ++i) {
      for (int j = 0; j < embed_dim; ++j) {
        uint32_t q = float_check[i * embed_dim + j];
        q <<= 16;
        float f = reinterpret_cast<float&>(q);
        printf("%0.4f ", f);
      }
      printf("\n");
    }
    delete [] float_check;
  }

  convertPCMtoTCM((uint16_t*)attn_accumulator.getHostAddr(), attn_output,
                  seq_len, embed_dim, batch_size, PROSE_Nmin);
  // memcpy_bf16_to_fp32(attn_output, (uint16_t*)attn_accumulator.getHostAddr(),
  // batch_size * seq_len * embed_dim);
}

void prose_float_wrapper::prose_matadd(float* input1, float* input2,
                                       float* output, uint32_t eles) {
  auto input1_ptr = handle.malloc(2 * eles);
  auto input2_ptr = handle.malloc(2 * eles);
  auto output_ptr = handle.malloc(2 * eles);

  memcpy_fp32_to_bf16((uint16_t*)input1_ptr.getHostAddr(), input1, eles);
  memcpy_fp32_to_bf16((uint16_t*)input2_ptr.getHostAddr(), input2, eles);

  prose_impl::prose_matadd(input1_ptr, input2_ptr, output_ptr, eles);

  handle.copy_from_fpga(output_ptr);
  memcpy_bf16_to_fp32(output, (uint16_t*)output_ptr.getHostAddr(), eles);
}

const int mlp_m = 3072;

void prose_float_wrapper::prose_mlp(float* input, float* fc_wgt, float* fc_bias,
                                    float* proj_wgt, float* proj_bias,
                                    float* out, int batch_size, int seq_len,
                                    int embed_size) {
  auto interm = handle.malloc(2 * batch_size * seq_len * mlp_m);
  prose_float_wrapper::prose_g_matmul(input, fc_wgt, interm, fc_bias, nullptr,
                                      PROSE_biasCOLS, batch_size, seq_len,
                                      embed_size, mlp_m, false);
  prose_float_wrapper::prose_m_matmul(interm, proj_wgt, out, proj_bias, nullptr,
                                      batch_size, seq_len, mlp_m, embed_size,
                                      true, false, false, PROSE_biasCOLS);
}

static void print_a_bit(float *f) {
  int l = 768;
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) {
      printf("%0.4f\t", f[i * l + j]);
    }
    printf("\n");
  }
}


void prose_float_wrapper::decoder_layer(
    float* input, float* ln1_wgt, float* ln1_bias, float* ln2_wgt,
    float* ln2_bias, float** qarray, float** karray, float** varray,
    float** oarray, float* obias, float* causal_mask, float* mlp_fc_wgt,
    float* mlp_fc_bias, float* mlp_proj_wgt, float* mlp_proj_bias,
    uint8_t batch_size, uint16_t seq_len, uint16_t embed_dim,
    uint16_t num_heads, float* attn_output) {
  float* interm = new float[batch_size * seq_len * embed_dim];
  float* interm2 = new float[batch_size * seq_len * embed_dim];
  float* residual = new float[batch_size * seq_len * embed_dim];

  memset(interm2, 0, sizeof(float) * batch_size * seq_len * embed_dim);
  prose_float_wrapper::prose_layer_norm(input, ln1_wgt, ln1_bias, batch_size,
                                        seq_len, embed_dim, interm);

  // attn
  prose_float_wrapper::prose_multi_head_attention(
      interm, qarray, karray, varray, oarray, obias, causal_mask, batch_size,
      seq_len, embed_dim, num_heads, interm2, false);

  std::cout << "ATTENTION OUT FINAL: " << std::endl;
  for (int i = 0; i < seq_len; ++i) {
    for (int j = 0; j < embed_dim; ++j) {
      float q = interm2[i * embed_dim + j];
      printf("%0.4f ", q);
    }
    printf("\n");
  }


  // add inputs to the attn output, call this residual
  prose_float_wrapper::prose_matadd(interm2, input, residual,
                                    batch_size * seq_len * embed_dim);
  std::cout << "AFTER ADD RESIDUAL: " << std::endl;
  for (int i = 0; i < seq_len; ++i) {
    for (int j = 0; j < embed_dim; ++j) {
      float q = residual[i * embed_dim + j];
      printf("%0.4f ", q);
    }
    printf("\n");
  }


  prose_float_wrapper::prose_layer_norm(residual, ln2_wgt, ln2_bias, batch_size,
                                        seq_len, embed_dim, interm2);

  std::cout << "AFTER ADD LN2: " << std::endl;
  for (int i = 0; i < seq_len; ++i) {
    for (int j = 0; j < embed_dim; ++j) {
      float q = interm2[i * embed_dim + j];
      printf("%0.4f ", q);
    }
    printf("\n");
  }

  // THESE QUALIFIER ARE NOT REDUNDANT
  prose_float_wrapper::prose_mlp(interm2, mlp_fc_wgt, mlp_fc_bias, mlp_proj_wgt,
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


  prose_float_wrapper::prose_matadd(residual, interm, attn_output,
                                    embed_dim * batch_size * seq_len);
  std::cout << "AFTER FINAL RESIDUAL ADD: " << std::endl;
  for (int i = 0; i < seq_len; ++i) {
    for (int j = 0; j < embed_dim; ++j) {
      float q = attn_output[i * embed_dim + j];
      printf("%0.4f ", q);
    }
    printf("\n");
  }
}
