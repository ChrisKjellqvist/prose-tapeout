
//
// Created by Christopher Kjellqvist on 9/30/24.
//

#include "auto_allocate.h"
#include "beethoven/fpga_handle.h"
#include "beethoven/rocc_cmd.h"
#include "beethoven_hardware.h"
#include "prose_rptr.h"
#include "prose_vec_rptr.h"
#include <coroutine>
#include <vector>

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
// Forward declarations for types that would be defined elsewhere
struct ModelConfig {
    int batch_size;
    int seq_len;
    int D;
    int head_size;
};
struct TransformerLayer {};
struct AllLayers {
    TransformerLayer layers[1];
};
template<int, int, int, int, int>
struct prose_allocations {
    remote_ptr selfatten_intermediates[1][4];
    remote_ptr selfatten_attenscore[1];
    remote_ptr ln_out[1];
    remote_ptr mlp_intermediate[1];
};


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
    static constexpr int NUM_HEAD = 12;

    unsigned char bits[TOTAL_BYTES] = {}; // array to store heads + decoder flags
    unsigned char heads[HEADS] = {}; // array to store heads + decoder flags
    unsigned short int mha_done = ~((1u << NUM_HEAD) - 1);

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


struct promise;
struct prose_thread : std::coroutine_handle<promise>
{
    using promise_type = ::promise;
    bool done() const {
        return this->promise().task_done;
    }
};

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

// Forward declarations of the coroutine functions
prose_thread prose_e_matmul(remote_ptr const &activations, remote_ptr const &weights,
                    remote_ptr const &out, remote_ptr const *bias,
                    remote_ptr const *norms, int biasMode,
                    int chosen_batch_size, bool weights_are_batched, int M,
                    int K, int N, remote_ptr const &write_out,
                    bool norm_per_batch, dec_dep* dep, const int head_id);

prose_thread prose_m_matmul(const remote_ptr &activations, const remote_ptr &weights,
                    const remote_ptr &out, remote_ptr const *bias, int biasMode,
                    int chosen_batch_size, int M, int K, int N,
                    bool output_transpose, const remote_ptr *norms,
                    bool weights_are_batched, bool norm_per_batch,
                    int stripe_stride, M_type which_m, dec_dep* dep, const int head_id);

prose_thread prose_g_matmul(remote_ptr const &activations, remote_ptr const &weights,
                    remote_ptr const *norms, remote_ptr const *bias,
                    remote_ptr const &out, int chosen_batch_size, int M, int K,
                    int N, bool norm_per_batch, int biasMode, dec_dep* dep);

prose_thread prose_layer_norm(const beethoven::remote_ptr &input,
                      const beethoven::remote_ptr &gamma_beta,
                      const uint8_t &batch_size, const uint16_t &input_length,
                      const uint16_t &seq_len,
                      const beethoven::remote_ptr &out, dec_dep* dep, DecoderMask which_norm);

prose_thread prose_matadd(const beethoven::remote_ptr &a,
                  const beethoven::remote_ptr &b,
                  const beethoven::remote_ptr &c, const uint32_t &length, dec_dep* dep, DecoderMask which_add);


// --- SCHEDULER ---
constexpr int MAX_TASKS = 100;
prose_thread tasks[MAX_TASKS];
int num_tasks = 0;

bool done() {
    for (int i = 0; i < num_tasks; i++) {
        if (tasks[i] && !tasks[i].done()) {
            return false;
        }
    }
    return true;
}

void scheduler() {
    while (num_tasks > 0) {
        prose_thread current_tasks[MAX_TASKS];
        int current_num_tasks = num_tasks;
        memcpy(current_tasks, tasks, num_tasks * sizeof(prose_thread));
        num_tasks = 0;

        for (int i = 0; i < current_num_tasks; ++i) {
            auto& task = current_tasks[i];
            if (task && !task.done()) {
                task.resume();
                if (!task.done()) {
                    tasks[num_tasks++] = task;
                }
            }
        }
    }
}

// --- TASK LAUNCHER ---
void setup_decoder_tasks(std::vector<prose_thread>& tasks, const remote_ptr &input, const ModelConfig &config,
                         const remote_ptr &out, int t_id, int layer_id, dec_dep* dep) {

    const auto &residual = input;

    // 1. First Layer Normalization
    tasks.push_back(prose_layer_norm(input, all_layers.layers[layer_id].ln1_wb, config.batch_size,
                     config.D, config.seq_len,
                     my_prose_allocations.ln_out[t_id], dep, norm1));

    const remote_ptr &attention_input = my_prose_allocations.ln_out[t_id];
    const remote_ptr(&temps)[4] = my_prose_allocations.selfatten_intermediates[t_id];
    auto &attention_score_matrix_temp = my_prose_allocations.selfatten_attenscore[t_id];
    const TransformerLayer &layer = all_layers.layers[layer_id];

    // Multi-Head Attention Loop
    for (int head_idx = 0; head_idx < config.head_size; ++head_idx) {
        tasks.push_back(prose_m_matmul(attention_input, layer.proj_wgts[head_idx].qproj, temps[0], nullptr,
                       PROSE_biasNONE, config.batch_size, config.seq_len, config.D,
                       config.head_size, true, nullptr, false, false, 0, M_type::M_Q, dep, head_idx));
        
        tasks.push_back(prose_m_matmul(attention_input, layer.proj_wgts[head_idx].kproj, temps[1], nullptr,
                       PROSE_biasNONE, config.batch_size, config.seq_len, config.D,
                       config.head_size, false, nullptr, false, false, 0, M_type::M_K, dep, head_idx));

        tasks.push_back(prose_e_matmul(temps[0], temps[1], attention_score_matrix_temp,
                       &all_layers.layers[layer_id].causal_mask, nullptr,
                       PROSE_biasMATRIX, config.batch_size, true, config.seq_len,
                       config.head_size, config.seq_len, temps[3], false, dep, head_idx));

        tasks.push_back(prose_m_matmul(attention_input, layer.proj_wgts[head_idx].vproj, temps[2], nullptr,
                       PROSE_biasNONE, config.batch_size, config.seq_len, config.D,
                       config.head_size, true, nullptr, false, false, 0, M_type::M_V, dep, head_idx));

        tasks.push_back(prose_m_matmul(attention_score_matrix_temp, temps[2], temps[1], nullptr,
                       PROSE_biasNONE, config.batch_size, config.seq_len,
                       config.seq_len, config.head_size, true, &temps[3],
                       true, true, 12, M_type::M_ATTN, dep, head_idx));
    }

    tasks.push_back(prose_m_matmul(temps[1], layer.oproj_w, out, &layer.oproj_b, PROSE_biasCOLS,
                 config.batch_size, config.seq_len, config.head_size, config.D,
                 true, nullptr, false, false, 0, M_type::M_FINAL, dep, 0));

    tasks.push_back(prose_matadd(residual, out, residual,
                    config.D * config.batch_size * config.seq_len, dep, add1));

    tasks.push_back(prose_layer_norm(residual, all_layers.layers[layer_id].ln2_wb, config.batch_size,
                   config.D, config.seq_len,
                   my_prose_allocations.ln_out[t_id], dep, norm2));

    tasks.push_back(prose_g_matmul(my_prose_allocations.ln_out[t_id],
                 all_layers.layers[layer_id].mlp_fc_w, nullptr,
                 &all_layers.layers[layer_id].mlp_fc_b,
                 my_prose_allocations.mlp_intermediate[t_id], config.batch_size,
                 config.seq_len, config.D, config.D * 4, 0, PROSE_biasCOLS, dep));

    tasks.push_back(prose_m_matmul(my_prose_allocations.mlp_intermediate[t_id],
                 all_layers.layers[layer_id].mlp_proj_w,
                 my_prose_allocations.ln_out[t_id],
                 &all_layers.layers[layer_id].mlp_proj_w, PROSE_biasCOLS,
                 config.batch_size, config.seq_len, config.D * 4, config.D,
                 true, nullptr, false, false, 0, M_type::M_DECODER, dep, 0));

    tasks.push_back(prose_matadd(residual, my_prose_allocations.ln_out[t_id],
                    out,
                    config.D * config.batch_size * config.seq_len, dep, add2));
}


// --- EXAMPLE MAIN ---
int main() {
    // This is a placeholder for your application's entry point.
    // You would initialize your hardware, load models, etc., here.

    // Example configuration
    ModelConfig config = {1, 16, 768, 12};
    remote_ptr input, out; // Assume these are initialized

    // Create a dependency tracker for the decoder layer
    dec_dep dependencies;

    // Launch all the tasks for the decoder layer.
    setup_decoder_tasks(tasks, input, config, out, 0, 0, &dependencies);

    // Run the scheduler to execute all the tasks.
    scheduler();

    return 0;
}

// NOTE: The actual implementations of the prose_... coroutine functions
// would need to be included in this file or a linked library.
