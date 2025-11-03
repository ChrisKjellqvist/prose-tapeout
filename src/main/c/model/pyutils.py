import torch

# assume 2-bytes per element
total_tensor_size_bytes = 0
layer_dict = {}
save = False
layer_save_limit = 1e9

def save_tensor(tensor, dirname, layername, layer_idx):
    global layer_save_limit
    global save
    if layer_idx >= layer_save_limit or not save:
        return
    if tensor.dtype == torch.bool:
        tensor = tensor.to(torch.float32)
        tensor = (tensor - 1) * torch.inf
    # print("trying to save " + layername)
    
    # save file
    fname = f"{dirname}/{layername}.pt"
    if len(tensor.shape) == 2:
        print(layername, len(tensor.T.shape), tensor.T.shape)
        torch.save(tensor.T.contiguous(), fname)
        layer_dict[layername] = tensor.T.shape
    else:
        print(layername, len(tensor.shape), tensor.shape)
        torch.save(tensor.contiguous(), fname)
        layer_dict[layername] = tensor.shape

    # store the names of the layers and their dimensions in a separate text file
    
    # update counter for total file size
    global total_tensor_size_bytes
    total_tensor_size_bytes += tensor.numel() * 2
    
def save_tensor_spl(tensor, heads, head_size, dirname, layername, layer_idx):
    global layer_save_limit
    global save
    if layer_idx >= layer_save_limit or not save:
        return
    # print("trying to save " + layername)
    
    # save file
    for i in range(heads):
        fname = f"{dirname}/{layername}.h{i}.pt"
        stripe = tensor.T[:,i*head_size:(i+1)*head_size].contiguous()
        torch.save(stripe, fname)
        layer_dict[f"{layername}.h{i}"] = stripe.shape
        
    global total_tensor_size_bytes
    total_tensor_size_bytes += tensor.numel() * 2

def write_layer_dict(fname):
    global save
    if not save:
        return
    with open(fname, 'w') as f:
        for key, value in layer_dict.items():
        # split the value into a series of space separated numbers
            k = value
            if isinstance(k, dict):
                k = " ".join(str(x) for x in k.values())
            elif isinstance(k, tuple):
                k = " ".join(str(x) for x in k)
            elif isinstance(k, torch.Size):
                k = " ".join(str(x) for x in k)
            else:
                k = " ".join(str(x) for x in k)
            f.write(f"{key} {k}\n")
    global total_tensor_size_bytes
    print("Estimated model size is " + str(total_tensor_size_bytes / 1024 / 1024) + " MB if we were to convert to BF16")


def enable_saving():
    global save
    save = True

def set_layer_save_limit(i):
    global layer_save_limit
    layer_save_limit = i
    

