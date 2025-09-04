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

model_id = "EleutherAI/gpt-neo-125m"
model = GPTNeoForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16)
config = model.config
model = GPTNeoMLP(4*768, config)
model.eval()
state_dict = model.state_dict()

with torch.no_grad():
    sample_input = torch.randn(1, 2, 768)
    sample_output = model(sample_input)

state_dict['input_mlp'] = sample_input
state_dict['output_mlp'] = sample_output

print(state_dict.keys())

for name, param in state_dict.items():
    # save to model_ckpts/
    torch.save(param.cpu().detach(), f"model_ckpts/{name}.pt")
