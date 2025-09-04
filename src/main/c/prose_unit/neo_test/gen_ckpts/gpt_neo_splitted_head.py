import torch
import numpy as np
import torch.nn as nn
import os
from transformers import GPTNeoForCausalLM
from transformers import GPT2Tokenizer
import torch.nn.functional as F
from typing import Optional, Tuple, Union
from transformers.cache_utils import Cache, DynamicCache, StaticCache
from transformers.modeling_outputs import (BaseModelOutputWithPast,
                                           BaseModelOutputWithPastAndCrossAttentions,
                                           CausalLMOutputWithCrossAttentions,
                                           CausalLMOutputWithPast,
                                           QuestionAnsweringModelOutput,
                                           SequenceClassifierOutputWithPast,
                                           TokenClassifierOutput, )

class GPTNeoSelfAttention(nn.Module):
    def __init__(self, config, attention_type):
        super().__init__()
        self.config = config

        max_positions = config.max_position_embeddings
        bias = torch.tril(torch.ones((max_positions, max_positions), dtype=bool)).view(
            1, 1, max_positions, max_positions
        )

        self.register_buffer("bias", bias, persistent=False)
        self.register_buffer("masked_bias", torch.tensor(-1e9), persistent=False)

        self.is_causal = True

        self.embed_dim = config.hidden_size
        self.num_heads = config.num_heads
        self.head_dim = self.embed_dim // self.num_heads
        if self.head_dim * self.num_heads != self.embed_dim:
            raise ValueError(
                f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim} and `num_heads`:"
                f" {self.num_heads})."
            )

        self.k_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=False)
        self.v_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=False)
        self.q_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=False)
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=True)

    def _split_heads(self, tensor, num_heads, attn_head_size):
        """
        Splits hidden_size dim into attn_head_size and num_heads
        """
        new_shape = tensor.size()[:-1] + (num_heads, attn_head_size)
        tensor = tensor.view(new_shape)
        return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)

    def _merge_heads(self, tensor, num_heads, attn_head_size):
        """
        Merges attn_head_size dim and num_attn_heads dim into hidden_size
        """
        tensor = tensor.permute(0, 2, 1, 3).contiguous()
        new_shape = tensor.size()[:-2] + (num_heads * attn_head_size,)
        return tensor.view(new_shape)

    def _attn(self, query, key, value, attention_mask=None, head_mask=None):
        # Keep the attention weights computation in fp32 to avoid overflow issues
        query = query.to(torch.float32)
        key = key.to(torch.float32)

        attn_weights = torch.matmul(query, key.transpose(-1, -2))

        query_length, key_length = query.size(-2), key.size(-2)
        causal_mask = self.bias[:, :, key_length - query_length: key_length, :key_length]
        mask_value = torch.finfo(attn_weights.dtype).min
        mask_value = torch.tensor(mask_value, dtype=attn_weights.dtype).to(attn_weights.device)
        attn_weights = torch.where(causal_mask, attn_weights, mask_value)

        if attention_mask is not None:
            # Apply the attention mask
            attn_weights = attn_weights + attention_mask

        attn_weights = nn.functional.softmax(attn_weights, dim=-1)
        attn_weights = attn_weights.to(value.dtype)

        # Mask heads if we want to
        if head_mask is not None:
            attn_weights = attn_weights * head_mask

        attn_output = torch.matmul(attn_weights, value)

        return attn_output, attn_weights

    def forward(
            self,
            hidden_states,
            attention_mask=None,
            layer_past=None,
            head_mask=None,
            use_cache=False,
            output_attentions=False,
    ):
        query = self.q_proj(hidden_states)
        key = self.k_proj(hidden_states)
        value = self.v_proj(hidden_states)

        # torch.save(query.float().cpu().detach(), "model_ckpts/attn_core.query.pt")
        # torch.save(key.float().cpu().detach(), "model_ckpts/attn_core.key.pt")
        # torch.save(value.float().cpu().detach(), "model_ckpts/attn_core.value.pt")

        query = self._split_heads(query, self.num_heads, self.head_dim)
        key = self._split_heads(key, self.num_heads, self.head_dim)
        value = self._split_heads(value, self.num_heads, self.head_dim)

        attn_output, attn_weights = self._attn(query, key, value, attention_mask, head_mask)

        attn_output = self._merge_heads(attn_output, self.num_heads, self.head_dim)

        attn_output = self.out_proj(attn_output)

        outputs = attn_output
        if output_attentions:
            outputs += (attn_weights,)

        return outputs  # a, present, (attentions)


def gpt_neo_self_attention_split_forward(
        hidden_states: torch.Tensor,
        q_proj_weight: torch.Tensor,
        k_proj_weight: torch.Tensor,
        v_proj_weight: torch.Tensor,
        out_proj_weight: torch.Tensor,
        out_proj_bias: torch.Tensor,
        num_heads: int,
        head_dim: int
) -> torch.Tensor:
    """
    Performs split self-attention in a GPT-Neo-like architecture using torch.matmul instead of F.linear.
    
    Args:
        hidden_states (torch.Tensor): Input tensor of shape (batch_size, seq_length, hidden_size).
        q_proj_weight (torch.Tensor): Query projection weights of shape (hidden_size, hidden_size).
        k_proj_weight (torch.Tensor): Key projection weights of shape (hidden_size, hidden_size).
        v_proj_weight (torch.Tensor): Value projection weights of shape (hidden_size, hidden_size).
        out_proj_weight (torch.Tensor): Output projection weights of shape (hidden_size, hidden_size).
        out_proj_bias (torch.Tensor): Output projection bias of shape (hidden_size,).
        num_heads (int): Number of attention heads.
        head_dim (int): Dimension of each attention head.
        attention_mask (torch.Tensor, optional): Additional attention mask. 
            Shape should be broadcastable to (batch_size, seq_length, seq_length).

    Returns:
        torch.Tensor: The output of the self-attention mechanism of shape (batch_size, seq_length, hidden_size).
    """
    batch_size, seq_length, embed_dim = hidden_states.size()
    expected_embed_dim = num_heads * head_dim

    if embed_dim != expected_embed_dim:
        raise ValueError(
            f"embed_dim must be equal to num_heads * head_dim (got embed_dim={embed_dim}, "
            f"num_heads={num_heads}, head_dim={head_dim})"
        )

    # Initialize the accumulator for attention outputs
    attn_output = torch.zeros(
        batch_size, seq_length, embed_dim, device=hidden_states.device, dtype=hidden_states.dtype
    )

    # Define mask value for positions that should not be attended to
    mask_value = torch.finfo(hidden_states.dtype).min
    torch.save(hidden_states, "model_ckpts/attn.input.pt")
    torch.save(q_proj_weight.T, "model_ckpts/attn.q_proj.weight.pt")
    torch.save(k_proj_weight.T, "model_ckpts/attn.k_proj.weight.pt")
    torch.save(v_proj_weight.T, "model_ckpts/attn.v_proj.weight.pt")
    torch.save(out_proj_weight.contiguous(), "model_ckpts/attn.o_proj.weight.pt")
    torch.save(out_proj_bias.contiguous(), "model_ckpts/attn.o_proj.bias.pt")

    for h in range(num_heads):
        q_proj_weight_h = q_proj_weight[h * head_dim: (h + 1) * head_dim, :]  # Shape: (head_dim, embed_dim)
        k_proj_weight_h = k_proj_weight[h * head_dim: (h + 1) * head_dim, :]  # Shape: (head_dim, embed_dim)
        v_proj_weight_h = v_proj_weight[h * head_dim: (h + 1) * head_dim, :]  # Shape: (head_dim, embed_dim)
        out_proj_weight_h = out_proj_weight[:, h * head_dim: (h + 1) * head_dim]  # Shape: (embed_dim, head_dim)
        # torch.save(q_proj_weight_h, f"model_ckpts/attn.h{h}.q_proj.weight.pt")
        # torch.save(k_proj_weight_h, f"model_ckpts/attn.h{h}.k_proj.weight.pt")
        # torch.save(v_proj_weight_h, f"model_ckpts/attn.h{h}.v_proj.weight.pt")

        # print("qproj_w\n", q_proj_weight_h.shape, "\n", q_proj_weight_h.T)

        # Compute query, key, value for the current head
        # Using torch.matmul: (batch_size, seq_length, embed_dim) @ (embed_dim, head_dim).T -> (batch_size, seq_length, head_dim)
        query_h = torch.matmul(hidden_states,
                               q_proj_weight_h.T)  # Equivalent to F.linear(hidden_states, q_proj_weight_h, bias=None)
        # print("qproj_res\n", query_h)
        key_h = torch.matmul(hidden_states,
                             k_proj_weight_h.T)  # Equivalent to F.linear(hidden_states, k_proj_weight_h, bias=None)
        # print("kproj_wgt\n", k_proj_weight_h.T)
        # print("kproj_res\n", key_h)
        value_h = torch.matmul(hidden_states,
                               v_proj_weight_h.T)  # Equivalent to F.linear(hidden_states, v_proj_weight_h, bias=None)

        # Compute attention scores
        # Shape: (batch_size, seq_length, seq_length)
        attn_scores = torch.matmul(query_h, key_h.transpose(-1, -2))  # (batch_size, seq_length, seq_length)
        # print("attn scores ", h)
        print(attn_scores)
        # Create causal mask
        causal_mask = torch.tril(torch.ones((seq_length, seq_length), dtype=torch.bool, device=hidden_states.device))
        # Expand mask to match batch size
        causal_mask = causal_mask.unsqueeze(0).expand(batch_size, -1, -1)  # Shape: (batch_size, seq_length, seq_length)
        causal_mask = (~causal_mask).float() * mask_value
        torch.save(causal_mask.cpu().detach(), "model_ckpts/attn.causal_mask.pt")

        print("adjusted attn:\n", torch.exp(attn_scores / np.sqrt(seq_length) + causal_mask))

        # Apply the causal mask: set scores of positions that should not be attended to to -inf
        attn_scores = attn_scores + causal_mask
        # Compute attention probabilities
        print("softmax in:\n", attn_scores)
        attn_probs = F.softmax(attn_scores, dim=-1)  # Shape: (batch_size, seq_length, seq_length)
        attn_probs = attn_probs.type_as(value_h)  # Ensure the same dtype as value_h
        print("softmaxed:\n", attn_probs)

        # print("vproj_weight\n", v_proj_weight_h.T)
        # print("vproj_out\n", value_h)
        # Compute the attention output
        attn_output_h = torch.matmul(attn_probs, value_h)  # Shape: (batch_size, seq_length, head_dim)
        # print("attn_final_out\n", attn_output_h)
        # print(attn_output_h)
        # print(out_proj_weight_h)
        # exit()

        # Apply the output projection using torch.matmul: (batch_size, seq_length, head_dim) @ (embed_dim, head_dim).T -> (batch_size, seq_length, embed_dim)
        output_h = torch.matmul(attn_output_h,
                                out_proj_weight_h.T)  # Equivalent to F.linear(attn_output_h, out_proj_weight_h, bias=None)
        # print("oproj_w\n", out_proj_weight_h.T)
        # print("oproj\n", output_h)

        # Accumulate the output
        attn_output += output_h

    # Add the output projection bias
    attn_output += out_proj_bias
    # print("attn_output\n", attn_output)
    torch.save(attn_output, "model_ckpts/attn.output.pt")
    return attn_output


class Custom_GPTNeoEmbedder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.wte = nn.Embedding(config.vocab_size, self.embed_dim)
        self.wpe = nn.Embedding(config.max_position_embeddings, self.embed_dim)

    def forward(
            self,
            input_ids: Optional[torch.Tensor] = None,
            past_key_values: Optional[Union[Cache, Tuple[torch.FloatTensor]]] = None,
            position_ids: Optional[torch.Tensor] = None,
            inputs_embeds: Optional[torch.Tensor] = None,
            use_cache: Optional[bool] = None,
            cache_position: Optional[torch.LongTensor] = None,
    ) -> Union[Tuple[torch.Tensor], BaseModelOutputWithPastAndCrossAttentions]:
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.wte(input_ids)

        # kept for BC (non `Cache` `past_key_values` inputs)
        if use_cache and not isinstance(past_key_values, Cache):
            if past_key_values is None:
                past_key_values = DynamicCache()
            else:
                past_key_values = DynamicCache.from_legacy_cache(past_key_values)
                print("AGH2")

        seq_length = inputs_embeds.shape[1]
        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(past_seen_tokens, past_seen_tokens + seq_length, device=inputs_embeds.device)

        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        position_embeds = self.wpe(position_ids)
        hidden_states = inputs_embeds + position_embeds
        return hidden_states


if __name__ == "__main__":
    torch.manual_seed(0xDEAD)
    model_id = "EleutherAI/gpt-neo-125m"
    model = GPTNeoForCausalLM.from_pretrained(model_id, torch_dtype=torch.float32)
    #### generate real inputs
    # print(model)
    tokenizer = GPT2Tokenizer.from_pretrained("EleutherAI/gpt-neo-125m", torch_dtype=torch.float32)
    prompt = ("Who am I?")
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids

    # print(input_ids)
    # print(model)

    ### RANDOMLY generate inputs
    # WE REQUIRE(!!!) this normalization otherwise the constants will go to inf.
    # input_ids = torch.randn((1, 4, 768)) / 16
    config = model.config
    model_golden = GPTNeoSelfAttention(config, attention_type=None)
    state_dict = model.state_dict()

    embedder = Custom_GPTNeoEmbedder(model.config)
    embedder.load_state_dict({
        "wte.weight": state_dict["transformer.wte.weight"],
        "wpe.weight": state_dict["transformer.wpe.weight"]
    })

    input_ids = embedder(input_ids)

    print("Who am I consists of ", input_ids.size(), " shape")

    # model_split = GPTNeoSelfAttentionSplit(config, attention_type=None)
    single_layer_dict = {
        "k_proj.weight": state_dict["transformer.h.0.attn.attention.k_proj.weight"],
        "q_proj.weight": state_dict["transformer.h.0.attn.attention.q_proj.weight"],
        "v_proj.weight": state_dict["transformer.h.0.attn.attention.v_proj.weight"],
        "out_proj.weight": state_dict["transformer.h.0.attn.attention.out_proj.weight"],
        "out_proj.bias": state_dict["transformer.h.0.attn.attention.out_proj.bias"],
    }
    model_golden.load_state_dict(single_layer_dict)
    # model_split.load_state_dict(state_dict)
    output_golden = model_golden(input_ids)
    # output_split = model_split(sample_input)

    # print("Ground Truth Output: ")
    # print(output_golden.shape)
    # print(output_golden)

    q_proj_weight = single_layer_dict["q_proj.weight"]  # Shape: (hidden_size, hidden_size)
    k_proj_weight = single_layer_dict["k_proj.weight"]  # Shape: (hidden_size, hidden_size)
    v_proj_weight = single_layer_dict["v_proj.weight"]  # Shape: (hidden_size, hidden_size)
    out_proj_weight = single_layer_dict["out_proj.weight"]  # Shape: (hidden_size, hidden_size)
    out_proj_bias = single_layer_dict["out_proj.bias"]  # Shape: (hidden_size,)

    # Initialize attention masks if necessary (here, using causal mask)
    # Create a causal mask (lower triangular matrix)
    seq_length = input_ids.size(1)
    # causal_mask = torch.tril(torch.ones((seq_length, seq_length), dtype=torch.bool))
    # Expand mask to match batch size
    batch_size = input_ids.size(0)
    # attention_mask = causal_mask.unsqueeze(0).expand(batch_size, -1, -1)  # Shape: (batch_size, seq_length, seq_length)
    # Convert mask to float: 0.0 for allowed positions, -inf for masked positions
    # attention_mask = attention_mask.float()
    # attention_mask = attention_mask.masked_fill(~attention_mask.bool(), torch.finfo(torch.float32).min)

    # Call the split attention function
    num_heads = 12
    head_dim = 64

    # torch.save(sample_input, "model_ckpts/attn.input.pt")
    # torch.save(q_proj_weight, "model_ckpts/attn.q_proj.pt")
    # torch.save(k_proj_weight, "model_ckpts/attn.k_proj.pt")
    # torch.save(v_proj_weight, "model_ckpts/attn.v_proj.pt")
    # torch.save(out_proj_weight.t().contiguous(), "model_ckpts/attn.output_proj_w.pt")
    # torch.save(sample_input, "model_ckpts/attn.output_proj_b.pt")

    output_split = gpt_neo_self_attention_split_forward(
        hidden_states=input_ids,
        q_proj_weight=q_proj_weight,
        k_proj_weight=k_proj_weight,
        v_proj_weight=v_proj_weight,
        out_proj_weight=out_proj_weight,
        out_proj_bias=out_proj_bias,
        num_heads=num_heads,
        head_dim=head_dim,
    )
    print("Split Head Output: ")
    print(output_split.shape)
    print(torch.max(torch.abs(output_split - output_golden)))

    # torch.save(output_split, "model_ckpts/attn.output.pt")
    print(output_split.shape)
