import torch
import torch.nn as nn
from transformers import GPTNeoForCausalLM


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

        self.attn_dropout = nn.Dropout(float(config.attention_dropout))
        self.resid_dropout = nn.Dropout(float(config.resid_dropout))
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
        causal_mask = self.bias[:, :, key_length - query_length : key_length, :key_length]
        mask_value = torch.finfo(attn_weights.dtype).min
        mask_value = torch.tensor(mask_value, dtype=attn_weights.dtype).to(attn_weights.device)
        attn_weights = torch.where(causal_mask, attn_weights, mask_value)

        if attention_mask is not None:
            # Apply the attention mask
            attn_weights = attn_weights + attention_mask

        attn_weights = nn.functional.softmax(attn_weights, dim=-1)
        attn_weights = attn_weights.to(value.dtype)
        attn_weights = self.attn_dropout(attn_weights)

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

        torch.save(query.float().cpu().detach(), "model_ckpts/attn_core.query.pt")
        torch.save(key.float().cpu().detach(), "model_ckpts/attn_core.key.pt")
        torch.save(value.float().cpu().detach(), "model_ckpts/attn_core.value.pt")

        query = self._split_heads(query, self.num_heads, self.head_dim)
        key = self._split_heads(key, self.num_heads, self.head_dim)
        value = self._split_heads(value, self.num_heads, self.head_dim)

        attn_output, attn_weights = self._attn(query, key, value, attention_mask, head_mask)

        attn_output = self._merge_heads(attn_output, self.num_heads, self.head_dim)
        torch.save(attn_output, "model_ckpts/attn_core.attn_output.pt")


        attn_output = self.out_proj(attn_output)
        attn_output = self.resid_dropout(attn_output)

        outputs = attn_output
        if output_attentions:
            outputs += (attn_weights,)

        out_t = attn_output.float().cpu().detach()
        print(out_t)


        return outputs  # a, present, (attentions)

def save_ckpts():
    model_id = "EleutherAI/gpt-neo-125m"
    model = GPTNeoForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16)
    config = model.config
    gpt_neo_self_attention = GPTNeoSelfAttention(config, attention_type=None)
    with torch.no_grad():
        sample_input = torch.randn(1, 4, 768)
        output = gpt_neo_self_attention(sample_input)
    state_dict = gpt_neo_self_attention.state_dict()
    state_dict['input'] = sample_input
    state_dict['output'] = output
    # state_dict['output'] = output[0]
    # save each tensor to the model ckpts
    for k, v in state_dict.items():
        torch.save(v.cpu().detach(), f"model_ckpts/attn.{k}.pt")
    torch.save(state_dict, "attn.state_dict.pt")
    exit()



def attn_core():
    def _split_heads(tensor):
        # group heads
        num_heads = 12
        attn_head_size = 64
        new_shape = tensor.size()[:-1] + (num_heads, attn_head_size)
        tensor = tensor.view(new_shape)
        return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    
    def _merge_heads(tensor):
        num_heads = 12
        attn_head_size = 64
        tensor = tensor.permute(0, 2, 1, 3).contiguous()
        new_shape = tensor.size()[:-2] + (num_heads * attn_head_size,)
        return tensor.view(new_shape)
        
    key = torch.load("model_ckpts/attn_core.key.pt")
    query = torch.load("model_ckpts/attn_core.query.pt")
    value = torch.load("model_ckpts/attn_core.value.pt")
    attn_output = torch.load("model_ckpts/attn_core.attn_output.pt")

    print("key:", key.shape)
    print("query:", query.shape)
    print("value:", value.shape)
    key = _split_heads(key)
    query = _split_heads(query)
    value = _split_heads(value)

    attn_weights = torch.matmul(query, key.transpose(-1, -2))

    # query_length, key_length = query.size(-2), key.size(-2)
    # causal_mask = self.bias[:, :, key_length - query_length : key_length, :key_length]
    # mask_value = torch.finfo(attn_weights.dtype).min
    # mask_value = torch.tensor(mask_value, dtype=attn_weights.dtype).to(attn_weights.device)
    # attn_weights = torch.where(causal_mask, attn_weights, mask_value)

    # if attention_mask is not None:
    #     # Apply the attention mask
    #     attn_weights = attn_weights + attention_mask

    attn_weights = nn.functional.softmax(attn_weights, dim=-1)

    attn_output = torch.matmul(attn_weights, value)

    attn_output = _merge_heads(attn_output)
    print("attn_output:", attn_output.shape)
    
    

if __name__ == "__main__":
    save_ckpts()
    attn_core()
