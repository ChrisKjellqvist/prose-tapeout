import torch
import torch.nn as nn
from transformers.models.gpt_neo.modeling_gpt_neo import ACT2FN 
from transformers import GPTNeoForCausalLM


class GPTNeoMLP(nn.Module):
    def __init__(self, intermediate_size, config):  # in MLP: intermediate_size= 4 * hidden_size
        super().__init__()
        embed_dim = config.hidden_size
        self.c_fc = nn.Linear(embed_dim, intermediate_size)
        self.c_proj = nn.Linear(intermediate_size, embed_dim)
        self.act = ACT2FN[config.activation_function]
        self.dropout = nn.Dropout(float(config.resid_dropout))

    def forward(self, hidden_states):
        hidden_states = self.c_fc(hidden_states)
        hidden_states = self.act(hidden_states)
        hidden_states = self.c_proj(hidden_states)
        hidden_states = self.dropout(hidden_states)
        return hidden_states


def test_golden():
    input_sample = torch.load("model_ckpts/input_mlp.pt")
    output_golden = torch.load("model_ckpts/output_mlp.pt")
    model_id = "EleutherAI/gpt-neo-125m"
    model = GPTNeoForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16)
    config = model.config
    model = GPTNeoMLP(4*768, config)
    model.eval()
    state_dict = model.state_dict()

    with torch.no_grad():
        test_output = model(input_sample)
        # print the output
        print(f"test_output: {test_output}")
        print(f"output_golden: {output_golden}")


def main():
    # Load tensors from .pt files
    c_fc_weight = torch.load("model_ckpts/c_fc.weight.pt")
    c_fc_bias = torch.load("model_ckpts/c_fc.bias.pt")
    c_proj_weight = torch.load("model_ckpts/c_proj.weight.pt")
    c_proj_bias = torch.load("model_ckpts/c_proj.bias.pt")
    input_tensor = torch.load("model_ckpts/input_mlp.pt")
    output_golden = torch.load("model_ckpts/output_mlp.pt")

    # Determine dimensions
    embedding_size = c_fc_weight.size(1)  # Assuming c_fc_weight shape is (768, 3072)
    intermediate_size = 4 * embedding_size  # Typically 3072 if embedding_size is 768
    batch_size = input_tensor.size(0)  # e.g., 1
    seq_len = input_tensor.size(1)     # e.g., 1024

    print(f"embedding_size: {embedding_size}")
    print(f"intermediate_size: {intermediate_size}")

    c_fc_out = torch.matmul(input_tensor, c_fc_weight.T) + c_fc_bias
    c_fc_out = torch.nn.functional.gelu(c_fc_out)
    print(c_fc_out.shape)
    print(c_fc_out)

    output = torch.matmul(c_fc_out, c_proj_weight.T) + c_proj_bias
    print(f"output: {output}")
    print(f"output_golden: {output_golden}")


if __name__ == "__main__":
    main()
    # test_golden()
