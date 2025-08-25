//
// Created by Entropy Xu on 1/29/24.
//

#ifndef C_EXEC_PYTORCH_DECODER_LAYER_GOLDEN_H
#define C_EXEC_PYTORCH_DECODER_LAYER_GOLDEN_H


#include <cstdint>
#include "pytorch_linear_golden.h"

namespace torch_model {
    struct decoder_layer_weights {
        uint16_t *k_proj_weight;
        uint16_t *k_proj_bias;
        uint16_t *q_proj_weight;
        uint16_t *q_proj_bias;
        uint16_t *v_proj_weight;
        uint16_t *v_proj_bias;
        uint16_t *out_proj_weight;
        uint16_t *out_proj_bias;
        uint16_t *up_proj_weight;
        uint16_t *up_proj_bias;
        uint16_t *down_proj_weight;
        uint16_t *down_proj_bias;
        uint16_t *pre_norm_weight;
        uint16_t *pre_norm_bias;
        uint16_t *after_norm_weight;
        uint16_t *after_norm_bias;
    };

    typedef struct decoder_layer_weights DecoderLayerWeights;

    void decoder_layer_bf16(uint16_t *input, uint16_t *output,
                       DecoderLayerWeights weights,
                       ActivationType activation_type,
                       unsigned batch_size, unsigned seq_len, unsigned head_num, unsigned head_size);
}

#endif //C_EXEC_PYTORCH_DECODER_LAYER_GOLDEN_H
