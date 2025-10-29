//
// Created by Christopher Kjellqvist on 9/30/24.
//

#include "auto_allocate.h"
#include "beethoven/fpga_handle.h"
#include "beethoven/rocc_cmd.h"
#include "beethoven_hardware.h"
#include "prose_rptr.h"
#include "prose_vec_rptr.h"
#include "prose_lib_twt.h"

using namespace beethoven;

#ifndef LOCAL
const constinit AllLayers all_layers;
const constinit prose_allocations<1, 768, 1, 16, 12> my_prose_allocations =
    auto_alloc::get_prose_allocs<1, 768, 1, 16, 12>();
#else
fpga_handle_t handle;
#include <sys/mman.h>
#include <unistd.h>
remote_ptr get_from_float_file(uint64_t offset, uint64_t len) {
  FILE *f = fopen("../../model/gpt_neo/prose_input.bin", "r");
  if (f == nullptr) {
    throw std::runtime_error("Cannot open ../../model/gpt_neo/prose_input.raw");
  }
  auto fptr = mmap(nullptr, len, PROT_READ, MAP_PRIVATE | MAP_FILE, fileno(f), offset);
  remote_ptr ptr = handle.malloc(len);
  memcpy(ptr.getHostAddr(), fptr, len);
  munmap(fptr, len);
  return ptr;
}
const AllLayers all_layers;
const prose_allocations<1, 768, 1, 16, 12> my_prose_allocations =
    auto_alloc::get_prose_allocs<1, 768, 1, 16, 12>();
#endif

enum class M_type {
  M_Q,
  M_K,
  M_V,
  M_ATTN,
  M_FINAL,
  M_DECODER
};
enum DecoderMask { norm1=0, mha=1, M1=2, norm2=3, add1=4, G=5, M2=6, add2=7 };
enum HeadMask { LeftE=0, RightE=1, LeftM=2, RightM=3 };


struct dec_dep {
    static constexpr int HEADS = 12;              // works for even number of heads
    static constexpr int BITS_PER_HEAD = 4;
    static constexpr int TOTAL_BITS_HEADS = HEADS * BITS_PER_HEAD;
    static constexpr int TOTAL_BYTES_HEADS = TOTAL_BITS_HEADS / 8; // exact bytes for heads
    static constexpr int DECODER_FLAGS = 8;      // 8 flags for decoder
    static constexpr int TOTAL_BYTES = TOTAL_BYTES_HEADS + 1; // last byte for decoder

    unsigned char bits[TOTAL_BYTES] = {}; // array to store heads + decoder flags
    unsigned char heads[HEADS] = {}; // array to store heads + decoder flags
    unsigned short int mha_done = ~((1u << HEADS) - 1);

    // HEAD flags
    void set_head(int head, HeadMask m) {
        int pos = head * BITS_PER_HEAD + m;
        int byte = pos / 8;
        int bit  = pos % 8;
        //if (value) 
        bits[byte] |= (1u << bit);
        //else       bits[byte] &= ~(1u << bit);
    }

    bool get_head(int head, HeadMask m) const {
        int pos = head * BITS_PER_HEAD + m;
        int byte = pos / 8;
        int bit  = pos % 8;
        return (bits[byte] >> bit) & 1u;
    }

    // DECODER flags (stored in last byte)

    void set_decoder(DecoderMask m) {
        //if (value) 
        bits[TOTAL_BYTES - 1] |= (1u << m);
        //else       bits[TOTAL_BYTES - 1] &= ~(1u << m);
    }

    bool get_decoder(DecoderMask m) const {
        return (bits[TOTAL_BYTES - 1] >> m) & 1u;
    }

    void set_head_done(int head_id) {
        mha_done|= (1u << head_id);
    }

    bool get_heads_done() {
        return mha_done == 0xFFFF;
    }

};



bool prose_thread::done() const noexcept {
    return this->promise().task_done;
}

void prose_thread::resume() noexcept {
    std::coroutine_handle<promise>::resume();
}
 
struct promise
{
    /** G E M Norm Add , 1: running, 0: idle */
    static bool prose_state[5];
    bool task_done = false;
    prose_thread get_return_object() { return {prose_thread::from_promise(*this)}; }
    std::suspend_always initial_suspend() noexcept { return {}; }
    std::suspend_never final_suspend() noexcept { return {}; }
    void return_value(bool v) {task_done = v;}
    void unhandled_exception() { task_done = false; }
};


struct decoder_scheduler {
  static constexpr int HEADS = 12; 
  prose_thread task_queue[HEADS * 5 + 7];
  dec_dep dependency;

  decoder_scheduler(const remote_ptr &input, const ModelConfig &config,
                            const remote_ptr &out, int t_id, int layer_id)
  {
    
    for (auto& task : task_queue) {
        task = nullptr;
    }
    set_decoder_tasks(input, config, out, t_id, layer_id);
  }

  void set_decoder_tasks(const remote_ptr &input, const ModelConfig &config,
                            const remote_ptr &out, int t_id, int layer_id);

  bool done() {
    for (const auto& task : task_queue) {
        if (task && !task.done()) {
          return false; // Found a valid task that is not yet done.
        }
    }
    return true; // All valid tasks are done, or the queue is empty.
  };
  
  void execute();
};


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
prose_thread prose_e_matmul_nb(remote_ptr const &activations, remote_ptr const &weights,
                    remote_ptr const &out, remote_ptr const *bias,
                    remote_ptr const *norms, int biasMode,
                    int chosen_batch_size, bool weights_are_batched, int M,
                    int K, int N, remote_ptr const &write_out,
                    bool norm_per_batch, const dec_dep* dep, const int head_id) { // using message passing, go if input pointer is true, write output point to true when finishing
  const auto activation_stripe_size_bytes =
      K * PROSE_Nmin * 2 * chosen_batch_size;
  auto output_stripe_sz_bytes = PROSE_ECore_N * N * 2 * chosen_batch_size;
  auto output_substripe_sz_bytes = PROSE_Nmin * N * 2 * chosen_batch_size;
  bool use_norms = norms != nullptr;
  bool dependency_solved = false;
  /** wait until dependancy is solved and core is free **/
  while (promise::prose_state[1] && !dependency_solved/* input dependency unsolved */)
  {
    dependency_solved = dep->get_head(head_id, LeftE) && dep->get_head(head_id, RightE);
    co_await std::suspend_always{};
  }
  /**  tiled matrix multiply **/
  // row tiles in output matrix
  int bias_addr_incr, bias_sz_bytes;
  if (biasMode == PROSE_biasNONE) {
  } else if (biasMode == PROSE_biasCOLS) {
    bias_sz_bytes = N * 2;
    bias_addr_incr = 0;
  } else if (biasMode == PROSE_biasMATRIX) {
    bias_sz_bytes = PROSE_Nmin * N * 2;
    bias_addr_incr = bias_sz_bytes * (PROSE_ECore_N / PROSE_Nmin);
  } else if (biasMode == PROSE_biasBATCHEDMATRIX) {
    bias_sz_bytes = PROSE_Nmin * N * chosen_batch_size * 2;
    bias_addr_incr = bias_sz_bytes * (PROSE_ECore_N / PROSE_Nmin);
  }
  int row_execs_to_do = M / PROSE_ECore_N;
  int cols_mo = N / PROSE_ECore_N - 1;
  auto act_acc = activations;
  auto out_acc = out;
  auto smax_acc = write_out;
  remote_ptr norm_acc;
  remote_ptr bias_acc;
  if (use_norms)
    norm_acc = *norms;
  if (biasMode != PROSE_biasNONE)
    bias_acc = *bias;
  auto handle = ECore::matrixOp(0, act_acc, weights, out_acc, chosen_batch_size, bias_acc,
                  biasMode, bias_sz_bytes, K, cols_mo, row_execs_to_do - 1,
                  norm_acc, use_norms, norm_per_batch,
                  output_substripe_sz_bytes, smax_acc, weights_are_batched); // not a bool, write a wrapper
  
  while(!handle.try_get().has_value())
  {
    co_await std::suspend_always();
  }
  dep->set_head(head_id, LeftM);
  co_return true;
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
prose_thread prose_m_matmul_nb(const remote_ptr &activations, const remote_ptr &weights,
                    const remote_ptr &out, remote_ptr const *bias, int biasMode,
                    int chosen_batch_size, int M, int K, int N,
                    bool output_transpose, const remote_ptr *norms,
                    bool weights_are_batched, bool norm_per_batch,
                    M_type which_m, dec_dep* dep, const int head_id, int stripe_stride) {
  const auto activation_stripe_size_bytes =
      K * PROSE_Nmin * 2 * chosen_batch_size;
  auto output_substripe_sz_bytes =
      (PROSE_Nmin * (output_transpose ? N : M) * 2 * chosen_batch_size) *
      stripe_stride;
  auto output_row_increment = 2 * chosen_batch_size * PROSE_MCore_N *
                              (output_transpose ? N : PROSE_Nmin);

  bool dependency_solved = false;

  while (promise::prose_state[2] && !dependency_solved)
  {
    switch (which_m) {
      case M_type::M_K:
        dependency_solved = dep->get_decoder(norm1);
        break;
      case M_type::M_Q:
        dependency_solved = dep->get_decoder(norm1);
      case M_type::M_V:
        dependency_solved = dep->get_decoder(norm1);
        break;
      case M_type::M_ATTN:
        dependency_solved = dep->get_head(head_id, LeftM) && dep->get_head(head_id, RightM);
        break;
      case M_type::M_FINAL:
        dependency_solved = dep->get_heads_done();
        break;
      case M_type::M_DECODER:
        dependency_solved =  dep->get_decoder(G);
        break;
    }
    co_await std::suspend_always();
  }
  /**  tiled matrix multiply **/
  // row tiles in output matrix
  
  bool use_norms = norms != nullptr;
  int bias_addr_incr, bias_sz_bytes;
  if (biasMode == PROSE_biasNONE) {
  } else if (biasMode == PROSE_biasCOLS) {
    bias_sz_bytes = N * 2;
    bias_addr_incr = 0;
  } else if (biasMode == PROSE_biasMATRIX) {
    bias_sz_bytes = PROSE_Nmin * N * 2;
    bias_addr_incr = bias_sz_bytes * (PROSE_MCore_N / PROSE_Nmin);
  } else if (biasMode == PROSE_biasBATCHEDMATRIX) {
    bias_sz_bytes = PROSE_Nmin * N * chosen_batch_size * 2;
    bias_addr_incr = bias_sz_bytes * (PROSE_MCore_N / PROSE_Nmin);
  }

  int rows_to_do = M / PROSE_MCore_N;
  int cols_mo = N / PROSE_MCore_N - 1;
  auto act_acc = activations;
  auto out_acc = out;
  remote_ptr norm_acc;
  remote_ptr bias_acc;
  if (use_norms)
    norm_acc = *norms;
  if (biasMode != PROSE_biasNONE)
    bias_acc = *bias;

  auto handle = MCore::matrixOp(0, act_acc, weights, out_acc, chosen_batch_size, bias_acc,
                  biasMode, bias_sz_bytes, K, cols_mo, rows_to_do - 1, norm_acc,
                  use_norms, norm_per_batch, output_substripe_sz_bytes,
                  output_transpose, weights_are_batched);

  while(!handle.try_get().has_value())
  {
    co_await std::suspend_always();
  }

  switch (which_m) {
      case M_type::M_K:
        dep->set_head(head_id, LeftE);
        break;
      case M_type::M_Q:
        dep->set_head(head_id, RightE);
      case M_type::M_V:
        dep->set_head(head_id, RightM);
        break;
      case M_type::M_ATTN:
        dep->set_head_done(head_id);
        break;
      case M_type::M_FINAL:
        dep->set_decoder(dep->get_decoder(mha));
        break;
      case M_type::M_DECODER:
        dependency_solved = dep->get_decoder(G);
        break;
  }
  co_return true;
}

prose_thread prose_g_matmul_nb(remote_ptr const &activations, remote_ptr const &weights,
                    remote_ptr const *norms, remote_ptr const *bias,
                    remote_ptr const &out, int chosen_batch_size, int M, int K,
                    int N, bool norm_per_batch, int biasMode, dec_dep* dep) {
  const auto activation_stripe_size_bytes =
      K * PROSE_Nmin * 2 * chosen_batch_size;
  const auto weight_stripe_size_bytes = K * PROSE_Nmin * 2;
  bool use_norms = norms != nullptr;

  auto output_stripe_sz_bytes = PROSE_GCore_N * N * 2 * chosen_batch_size;
  auto output_substripe_sz_bytes = PROSE_Nmin * N * 2 * chosen_batch_size;
  bool dependency_solved = dep->get_decoder(add1);
  while (promise::prose_state[0] && !dependency_solved)
  {
    dependency_solved = dep->get_decoder(add1);
    co_await std::suspend_always{};
  }
  /**  tiled matrix multiply **/
  // row tiles in output matrix
  int bias_addr_incr, bias_sz_bytes;
  if (biasMode == PROSE_biasNONE) {
  } else if (biasMode == PROSE_biasCOLS) {
    bias_sz_bytes = N * 2;
    bias_addr_incr = 0;
  } else if (biasMode == PROSE_biasMATRIX) {
    bias_sz_bytes = PROSE_Nmin * N * 2;
    bias_addr_incr = bias_sz_bytes * (PROSE_GCore_N / PROSE_Nmin);
  } else if (biasMode == PROSE_biasBATCHEDMATRIX) {
    bias_sz_bytes = PROSE_Nmin * N * chosen_batch_size * 2;
    bias_addr_incr = bias_sz_bytes * (PROSE_GCore_N / PROSE_Nmin);
  }

  int row_execs_to_do = M / PROSE_GCore_N;
  int cols_mo = N / PROSE_GCore_N - 1;

  auto act_acc = activations;
  auto out_acc = out;
  remote_ptr norm_acc;
  remote_ptr bias_acc;
  if (use_norms)
    norm_acc = *norms;
  if (biasMode != PROSE_biasNONE)
    bias_acc = *bias;

  auto handle = GCore::matrixOp(0, act_acc, weights, out_acc, chosen_batch_size, bias_acc,
                  biasMode, bias_sz_bytes, K, cols_mo, row_execs_to_do - 1,
                  norm_acc, use_norms, norm_per_batch,
                  output_substripe_sz_bytes, true, false);

  while (!handle.try_get().has_value())
  {
    co_await std::suspend_always{};
  }
  dep->set_decoder(G);
  co_return true;
}

prose_thread prose_layer_norm_nb(const beethoven::remote_ptr &input,
                      const beethoven::remote_ptr &gamma_beta,
                      const uint8_t &batch_size, const uint16_t &input_length,
                      const uint16_t &seq_len,
                      const beethoven::remote_ptr &out, dec_dep* dep, DecoderMask which_norm) {
  if(which_norm != norm1){
    bool dependency_solved = dep->get_decoder(add1);
    while (promise::prose_state[3] && !dependency_solved)
    {
      dependency_solved = dep->get_decoder(add1);
      co_await std::suspend_always{};
    }
    
  }
  float norm = 1.F / float(input_length);
  uint32_t norm_fp = reinterpret_cast<uint32_t &>(norm);
  auto handle = Norm::norm(0, gamma_beta, input, batch_size * PROSE_Nmin,
             (seq_len / PROSE_Nmin), norm_fp >> 16, flagLayerNorm, out, true,
             input_length);
  while(!handle.try_get().has_value())
  {
    co_await std::suspend_always();
  }
  dep->set_decoder(which_norm);
  co_return true;
}

prose_thread prose_matadd_nb(const beethoven::remote_ptr &a,
                  const beethoven::remote_ptr &b,
                  const beethoven::remote_ptr &c, const uint32_t &length, dec_dep* dep, DecoderMask which_add) {
  bool dependency_solved = which_add == add1 ? dep->get_decoder(norm2) : dep->get_decoder(M2);
  while (promise::prose_state[4] && !dependency_solved)
  {
    dependency_solved = dep->get_decoder(add1);
    co_await std::suspend_always{};
  }
  auto handle = MatrixAdd::MatAdd(0, a, b, c, length);
  while(!handle.try_get().has_value()){
    co_await std::suspend_always();
  }
  dep->set_decoder(which_add);
  co_return true;
}



/**
 * @brief decoder task scheduler logic
 *
 * This function pushes tasks in the granularity of PROSE OPs based on 'prose_mh_self_attention'
 * and push them into a queue directly (NON-blocking)
 */
void decoder_scheduler::set_decoder_tasks(const remote_ptr &input, const ModelConfig &config,
                            const remote_ptr &out, int t_id, int layer_id) {
  // The initial input is saved for the first residual connection.
  prose_thread* task_queue = this->task_queue;
  const auto &residual = input;

  // 1. First Layer Normalization
  // Norm::norm(0, all_layers.layers[layer_id].ln1_wb, input, 1, config.batch_size,
  //            1.0 / 768, flagLayerNorm, my_prose_allocations.ln_out[t_id], 1,
  //            config.D)
  //     .get();
  task_queue[0] = prose_layer_norm_nb(input, all_layers.layers[layer_id].ln1_wb, config.batch_size,
                   config.D, config.seq_len,
                   my_prose_allocations.ln_out[t_id], &this->dependency, norm1);

  // The 'input' to the original attention function is the output of the first LayerNorm.
  const remote_ptr &attention_input = my_prose_allocations.ln_out[t_id];

  // Set up local pointers to temporary buffers and layer weights, same as in the original function.
  const remote_ptr(&temps)[4] =
      my_prose_allocations.selfatten_intermediates[t_id];
  auto &attention_score_matrix_temp =
      my_prose_allocations.selfatten_attenscore[t_id];
  const TransformerLayer &layer = all_layers.layers[layer_id];

  // Multi-Head Attention Loop
  for (int head_idx = 0; head_idx < config.head_size; ++head_idx) {
    // QUERY PROJECTION
    task_queue[1 + 5 * head_idx] = prose_m_matmul_nb(attention_input, layer.proj_wgts[head_idx].qproj, temps[0], nullptr,
                   PROSE_biasNONE, config.batch_size, config.seq_len, config.D,
                   config.head_size, true, nullptr, false, false, M_type::M_Q, &this->dependency,head_idx);
    // KEY PROJECTION
    task_queue[2 + 5 * head_idx] = prose_m_matmul_nb(attention_input, layer.proj_wgts[head_idx].kproj, temps[1], nullptr,
                   PROSE_biasNONE, config.batch_size, config.seq_len, config.D,
                   config.head_size,
                   false /* DOUBLE CHECK: Use non-transpose output here*/,
                   nullptr, false, false, M_type::M_K, &this->dependency,head_idx);
    // SOFTMAX(QUERY X KEY^T)
    task_queue[3 + 5 * head_idx] = prose_e_matmul_nb(
        temps[0], temps[1], attention_score_matrix_temp,
        &all_layers.layers[layer_id].causal_mask,
        nullptr /* DOUBLE CHECK: GPTNeo doesn't use scaled attention*/,
        PROSE_biasMATRIX, config.batch_size, true, config.seq_len,
        config.head_size, config.seq_len, temps[3], false, &this->dependency, head_idx);

    // VALUE PROJECTION
    task_queue[4 + 5 * head_idx] = prose_m_matmul_nb(attention_input, layer.proj_wgts[head_idx].vproj, temps[2], nullptr,
                   PROSE_biasNONE, config.batch_size, config.seq_len, config.D,
                   config.head_size,
                   true, // transpose (yes)
                   nullptr,
                   false, // weights are batched (no)
                   false,  // norm-per-batch (no)
                   M_type::M_V,
                   &this->dependency,
                   head_idx
    );

    // ATTENTION OUTPUT (Scores * Values)
    task_queue[5 + 5 * head_idx] = prose_m_matmul_nb(attention_score_matrix_temp, temps[2], temps[1], nullptr,
                   PROSE_biasNONE, config.batch_size, config.seq_len,
                   config.seq_len, config.head_size, true, &temps[3],
                   true, // normalize per batch with softmax norm factors
                   true, // weights are batched
                   M_type::M_FINAL,
                   &this->dependency,
                   head_idx,
                   12    // stripe stride will hop across all of the other heads

    );
  }

  // Final Output Projection of Attention Heads. The result is written to 'out'.
  task_queue[config.head_size * 5 + 1] = prose_m_matmul_nb(temps[1], layer.oproj_w, out, &layer.oproj_b, PROSE_biasCOLS,
                 config.batch_size, config.seq_len, config.head_size, config.D,
                 true, nullptr, false, false, M_type::M_FINAL, &this->dependency, 0);
  // --- END INLINED prose_mh_self_attention ---

  // 2. First Residual Connection (Add & Norm)
  // Adds the original input to the output of the attention block.
  // MatrixAdd::MatAdd(0, residual, out, residual,
  //                   config.D * config.batch_size * config.seq_len)
  //     .get();
  task_queue[config.head_size * 5 + 2] = prose_matadd_nb(residual, out, residual, config.D * config.batch_size * config.seq_len, &this->dependency, add1);

  // 3. Second Layer Normalization
  // The result of the first residual connection is passed to the second norm.
  // Norm::norm(0, all_layers.layers[layer_id].ln2_wb, residual, 1,
  //            config.batch_size, 1.0 / 768, flagLayerNorm,
  //            my_prose_allocations.ln_out[t_id], 1, config.D)
  //     .get();
  task_queue[config.head_size * 5 + 3] = prose_layer_norm_nb(residual, all_layers.layers[layer_id].ln2_wb, config.batch_size,
                   config.D, config.seq_len,
                   my_prose_allocations.ln_out[t_id], &this->dependency, norm2);

  // 4. MLP (Feed-Forward Network)
  // The output of the second norm is the input to the MLP.
  task_queue[config.head_size * 5 + 4] = prose_g_matmul_nb(my_prose_allocations.ln_out[t_id],
                 all_layers.layers[layer_id].mlp_fc_w, nullptr,
                 &all_layers.layers[layer_id].mlp_fc_b,
                 my_prose_allocations.mlp_intermediate[t_id], config.batch_size,
                 config.seq_len, config.D, config.D * 4, 0, PROSE_biasCOLS, &this->dependency);

  task_queue[config.head_size * 5 + 5] = prose_m_matmul_nb(my_prose_allocations.mlp_intermediate[t_id],
                 all_layers.layers[layer_id].mlp_proj_w,
                 my_prose_allocations.ln_out[t_id],
                 &all_layers.layers[layer_id].mlp_proj_w, PROSE_biasCOLS,
                 config.batch_size, config.seq_len, config.D * 4, config.D,
                 true, nullptr, false, false, M_type::M_DECODER, &this->dependency, 0);

  // 5. Second Residual Connection
  // Adds the result of the first residual connection to the output of the MLP.
  // The final result of the decoder layer is written to 'out'.
  // MatrixAdd::MatAdd(0, residual, my_prose_allocations.ln_out[t_id],
  //                   out,
  //                   config.D * config.batch_size * config.seq_len)
  //     .get();
  task_queue[config.head_size * 5 + 6] = prose_matadd_nb(residual, my_prose_allocations.ln_out[t_id], out, config.D * config.batch_size * config.seq_len, &this->dependency, add2);
}


void decoder_scheduler::execute(){
 
  while(!this->done())
  {
    for(auto &i : this->task_queue)
    {
      i.resume();
    }
  }
}