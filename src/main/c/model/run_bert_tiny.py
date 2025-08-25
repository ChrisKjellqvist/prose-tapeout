# Load model directly
from transformers import AutoTokenizer, AutoModelForQuestionAnswering
import torch
import os

# save the weight matrix and bias vector of the first layer of the self attention into a raw array
import numpy as np

# assume 2-bytes per element
total_tensor_size_bytes:float = 0

def save_tensor(tensor, filename):
    np.save(filename, tensor.detach().numpy())
    global total_tensor_size_bytes
    sz = tensor.numel() * 2
    total_tensor_size_bytes += sz
    print(filename, " is ", sz, "B")

if __name__ == "__main__":
    tokenizer = AutoTokenizer.from_pretrained("Intel/dynamic_tinybert")
    model = AutoModelForQuestionAnswering.from_pretrained("Intel/dynamic_tinybert")

    def ask_question(question, context):
        inputs = tokenizer(question, context, return_tensors="pt")
        start_positions = torch.tensor([1])
        end_positions = torch.tensor([3])

        # print out the shape of the question and context matrix
        print(inputs["input_ids"].shape)
        print(inputs["attention_mask"].shape)


        outputs = model(**inputs, start_positions=start_positions, end_positions=end_positions)
        start_scores = outputs.start_logits
        end_scores = outputs.end_logits

        answer_start = torch.argmax(start_scores)
        answer_end = torch.argmax(end_scores) + 1

        return tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(inputs["input_ids"][0][answer_start:answer_end]))


    context = ("My name is Chris. I have three cats named Simon, Garfunkel, and Goobie. I went to undergrad at the University"
               "of Rochester for computer science. I am currently a PhD student at Duke University and doing research on"
               "hardware accelerators and associated frameworks. I am TAing an undergrad class right now and it is the"
               "absolute, absolute worst. Thank you for coming to my TED talk.")
    question = "What degree am I pursuing at Duke University?"

    # the first layer is an embedding layer
    # inside the embedding layer, the first layer is a token embedding layer
    # an embedding layer is a linear layer with a weight matrix and a bias vector
    # the dimension of the weight matrix is (vocab_size, hidden_size)
    # the second layer is a position embedding layer
    # a dropout layer is applied to the output of the embedding layer
    # is the dropout layer disabled in inference mode? (should be True)

    # then we move onto the encoder
    # the encoder is 6 layers deep of transformer blocks
    # in each transformer block, there is a self-attention layer and a feedforward layer
    # the self-attention layer is composed of a query, key, and value linear layer
    # the feedforward layer is composed of two linear layers
    # there is a residual connection around the self-attention and feedforward layers
    # there is a layer normalization layer around the residual connection
    # there is a dropout layer after the residual connection
    # is the dropout layer disabled in inference mode? (should be True)

    # then we move onto the output layer
    # the output layer is a linear layer with a weight matrix and a bias vector

    # for each layer, save it to a .pt file with a corresponding name and step.
    # each file should be name as {step}_{layer_name}.pt
    # matmuls and biases should be saved as separate files
    # the weight matrix should be saved as {step}_{layer_name}_weight.pt
    # the bias vector should be saved as {step}_{layer_name}_bias.pt

    dirname = "tinybert"
    os.makedirs(dirname, exist_ok=True)



    # for each layer in the model, save it with save_tensor and store the corresponding matrix size and file name in a dict
    layer_dict = {}
    for i, layer in enumerate(model.bert.encoder.layer):
        save_tensor(layer.attention.self.query.weight, f"{dirname}/{i}_self_query_weight.npy")
        save_tensor(layer.attention.self.query.bias, f"{dirname}/{i}_self_query_bias.npy")
        save_tensor(layer.attention.self.key.weight, f"{dirname}/{i}_self_key_weight.npy")
        save_tensor(layer.attention.self.key.bias, f"{dirname}/{i}_self_key_bias.npy")
        save_tensor(layer.attention.self.value.weight, f"{dirname}/{i}_self_value_weight.npy")
        save_tensor(layer.attention.self.value.bias, f"{dirname}/{i}_self_value_bias.npy")
        save_tensor(layer.attention.output.dense.weight, f"{dirname}/{i}_output_dense_weight.npy")
        save_tensor(layer.attention.output.dense.bias, f"{dirname}/{i}_output_dense_bias.npy")
        save_tensor(layer.attention.output.LayerNorm.weight, f"{dirname}/{i}_output_LayerNorm_weight.npy")
        save_tensor(layer.attention.output.LayerNorm.bias, f"{dirname}/{i}_output_LayerNorm_bias.npy")
        save_tensor(layer.intermediate.dense.weight, f"{dirname}/{i}_intermediate_dense_weight.npy")
        save_tensor(layer.intermediate.dense.bias, f"{dirname}/{i}_intermediate_dense_bias.npy")
        save_tensor(layer.output.dense.weight, f"{dirname}/{i}_output_dense_weight.npy")
        save_tensor(layer.output.dense.bias, f"{dirname}/{i}_output_dense_bias.npy")
        save_tensor(layer.output.LayerNorm.weight, f"{dirname}/{i}_output_LayerNorm_weight.npy")
        save_tensor(layer.output.LayerNorm.bias, f"{dirname}/{i}_output_LayerNorm_bias.npy")
        layer_dict.update({
            f"{i}_self_query_weight": layer.attention.self.query.weight.shape,
            f"{i}_self_query_bias": layer.attention.self.query.bias.shape,
            f"{i}_self_key_weight": layer.attention.self.key.weight.shape,
            f"{i}_self_key_bias": layer.attention.self.key.bias.shape,
            f"{i}_self_value_weight": layer.attention.self.value.weight.shape,
            f"{i}_self_value_bias": layer.attention.self.value.bias.shape,
            f"{i}_output_dense_weight": layer.attention.output.dense.weight.shape,
            f"{i}_output_dense_bias": layer.attention.output.dense.bias.shape,
            f"{i}_output_LayerNorm_weight": layer.attention.output.LayerNorm.weight.shape,
            f"{i}_output_LayerNorm_bias": layer.attention.output.LayerNorm.bias.shape,
            f"{i}_intermediate_dense_weight": layer.intermediate.dense.weight.shape,
            f"{i}_intermediate_dense_bias": layer.intermediate.dense.bias.shape
        })

    # do qa_outputs
    save_tensor(model.qa_outputs.weight, f"{dirname}/qa_outputs_weight.npy")
    save_tensor(model.qa_outputs.bias, f"{dirname}/qa_outputs_bias.npy")
    layer_dict.update({
        "qa_outputs_weight": model.qa_outputs.weight.shape,
        "qa_outputs_bias": model.qa_outputs.bias.shape
    })

    # save the layer_dict to a text file mapping file names to matrix sizes
    with open(f"{dirname}/layer_dict.txt", "w") as f:
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

    print(model)
