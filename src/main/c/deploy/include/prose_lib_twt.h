#ifndef PROSE_LIB_TWT_H
#define PROSE_LIB_TWT_H


#ifdef LOCAL
#include <beethoven/allocator/alloc.h>
#include <beethoven/fpga_handle.h>
#else
#include <beethoven_baremetal/allocator/alloc_baremetal.h>
#include <beethoven_baremetal/fpga_handle.h>
#endif
#include <beethoven_hardware.h>
#include <coroutine>
#include <prose_lib.h>
//
// Created by Christopher Kjellqvist on 9/30/24.
//

#include "beethoven/rocc_cmd.h"
using namespace beethoven;



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
    //unsigned char heads[HEADS] = {}; // array to store heads + decoder flags
    unsigned short int mha_done = ~((1u << HEADS) - 1);

    // HEAD flags
    void set_head(const int head, const HeadMask m) {
        int pos = head * BITS_PER_HEAD + m;
        int byte = pos / 8;
        int bit  = pos % 8;
        //if (value) 
        bits[byte] |= (1u << bit);
        //else       bits[byte] &= ~(1u << bit);
    }

    bool get_head(const int head, const HeadMask m) const {
        int pos = head * BITS_PER_HEAD + m;
        int byte = pos / 8;
        int bit  = pos % 8;
        return (bits[byte] >> bit) & 1u;
    }

    // DECODER flags (stored in last byte)

    void set_decoder(const int head_id) {
        //if (value) 
        bits[TOTAL_BYTES - 1] |= (1u << head_id);
        //else       bits[TOTAL_BYTES - 1] &= ~(1u << m);
    }

    bool get_decoder(const DecoderMask m) const {
        return (bits[TOTAL_BYTES - 1] >> m) & 1u;
    }

    void set_head_done(const int head_id) {
        mha_done|= (1u << head_id);
    }

    bool get_heads_done() {
        return mha_done == 0xFFFF;
    }

};

struct prose_promise;

struct prose_thread : std::coroutine_handle<prose_promise>
{
    using promise_type = ::prose_promise; 
    bool done() const noexcept;
    void resume() noexcept;
    const char* name() const noexcept;
};

struct prose_promise
{
  const char* name = __func__;
  bool task_done = false;
  prose_thread get_return_object() { return {prose_thread::from_promise(*this)}; }
  std::suspend_always initial_suspend() noexcept { return {}; }
  std::suspend_never final_suspend() noexcept { return {}; }
  void return_value(bool v) {task_done = v;}
  void unhandled_exception() { task_done = false; }
};


struct decoder_scheduler {
    /** G E M Norm Add , 1: running, 0: idle */
  static bool prose_state[5];
  static constexpr int HEADS = 12; 
  prose_thread task_queue[HEADS * 5 + 7];
  dec_dep dependency;

  decoder_scheduler(const remote_ptr &input, const ModelConfig &config,
                            const remote_ptr &out, int t_id, int layer_id);
  void set_decoder_tasks(const remote_ptr &input, const ModelConfig &config,
                            const remote_ptr &out, int t_id, int layer_id);
  bool done();
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
                    bool norm_per_batch, dec_dep* dep, const int head_id);
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
prose_thread prose_m_matmul_nb(remote_ptr const &activations, remote_ptr const &weights,
                    remote_ptr const &out, remote_ptr const *bias, int biasMode,
                    int chosen_batch_size, int M, int K, int N,
                    bool output_transpose, remote_ptr const *norms,
                    bool weights_are_batched, bool norm_per_batch, M_type which_m, dec_dep* dep, const int head_id,
                    int stripe_stride = 1);

prose_thread prose_g_matmul_nb(remote_ptr const &activations, remote_ptr const &weights,
                    remote_ptr const *norms, remote_ptr const *bias,
                    remote_ptr const &out, int chosen_batch_size, int M, int K,
                    int N, bool norm_per_batch, int biasMode, dec_dep* dep);

prose_thread prose_layer_norm_nb(const beethoven::remote_ptr &input,
                      const beethoven::remote_ptr &gamma_beta,
                      const uint8_t &batch_size, const uint16_t &input_length,
                      const uint16_t &seq_len,
                      const beethoven::remote_ptr &out, dec_dep* dep, DecoderMask which_norm);

prose_thread prose_matadd_nb(const beethoven::remote_ptr &a,
                  const beethoven::remote_ptr &b,
                  const beethoven::remote_ptr &c, const uint32_t &length,
                dec_dep* dep, DecoderMask which_add);


// prose_thread prose_decoder_nb(const remote_ptr &input, const ModelConfig &config,
//                    const remote_ptr &out_accumulation, int t_id, int layer_id, dec_dep* dep, DecoderMask which_norm);


#endif