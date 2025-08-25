import torch

state_dict = torch.load("attn.state_dict.pt")
print(state_dict.keys())
for k, v in state_dict.items():
    print(k, v.shape)
    torch.save(v.cpu().detach(), f"model_ckpts/attn.{k}.pt")
