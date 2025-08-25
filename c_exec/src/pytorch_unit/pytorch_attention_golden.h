//
// Created by Entropy Xu on 1/29/24.
//

#ifndef C_EXEC_PYTORCH_ATTENTION_GOLDEN_H
#define C_EXEC_PYTORCH_ATTENTION_GOLDEN_H

#include "torch/torch.h"
#include <cstdint>

namespace torch_model {
    void multihead_attention_bf16(uint16_t *K, uint16_t* Q, uint16_t* V, uint16_t* output,
                              unsigned batch_size, unsigned seq_len, unsigned head_num, unsigned head_size,
                              float scalar);
}


#endif //C_EXEC_PYTORCH_ATTENTION_GOLDEN_H
