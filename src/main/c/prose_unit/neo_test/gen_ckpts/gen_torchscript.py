import networkx as nx
import torch
from transformers import AutoModelForCausalLM
from torch.export import export
from pyvis.network import Network

# Load the model from the pre-trained identifier
model_id = "EleutherAI/gpt-neo-125m"
model = AutoModelForCausalLM.from_pretrained(model_id, device_map="cpu", torch_dtype=torch.float16, torchscript=True)

# Extract the first hidden layer from the transformer
one_hidden = model.transformer.h[0]

print(model)

# Example usage of scripted module
B = 4  # Batch size
L = 128  # Sequence length
D = 768  # Dimension size from model config
test_input = torch.randn(B, L, D, dtype=torch.float16)

# gm = torch.fx.symbolic_trace(one_hidden)
gm = export(one_hidden, (test_input,))
gm.graph.lint()
g = gm.graph
input_rename_dict = {}
for spec in gm.graph_signature.input_specs:
    real_name = spec.target
    node_name = spec.arg.name
    input_rename_dict[node_name] = real_name

g.print_tabular()

nx_graph = nx.DiGraph()
const_count = 0
for node in g.nodes:
    opcode = node.op
    if opcode == "const":
        continue
    name = node.name
    target = node.target
    args = node.args
    kwargs = node.kwargs
    # if has a member called "tensor_meta", then append it to target
    if node.meta.get("tensor_meta") is not None:
        meta = node.meta['tensor_meta'].shape
    else:
        meta = None
    nx_graph.add_node(name, opcode=str(opcode), target=str(target), meta=meta)
    if opcode == "output":
        nx_graph.add_edge(str(args[0][0]), name)
        continue
    for arg in args:
        # if arg is a node
        if isinstance(arg, torch.fx.Node):
            nx_graph.add_edge(arg.name, name)
        else:
            const_name = f"const_{const_count}"
            nx_graph.add_node(const_name, opcode="const", target=str(arg), args=[], kwargs={})
            nx_graph.add_edge(const_name, name)
            const_count += 1

# verify the graph is a DAG
assert nx.is_directed_acyclic_graph(nx_graph)

for nodes in nx_graph.nodes:
    # put const into group 2
    if nx_graph.nodes[nodes]["opcode"] == "const":
        nx_graph.nodes[nodes]["group"] = 0
        nx_graph.nodes[nodes]["size"] = 5
        label = f"{nx_graph.nodes[nodes]['target']}"
        nx_graph.nodes[nodes]["label"] = label
    # put input into group 1
    elif nx_graph.nodes[nodes]['opcode'] == 'placeholder':
        nx_graph.nodes[nodes]["group"] = 1
        nx_graph.nodes[nodes]["size"] = 10
        node_name = nx_graph.nodes[nodes]['target']
        label = f"{input_rename_dict[node_name]}"
        nx_graph.nodes[nodes]["label"] = label
    elif nx_graph.nodes[nodes]['opcode'] == 'output':
        nx_graph.nodes[nodes]["group"] = 2
        nx_graph.nodes[nodes]["size"] = 10
    else:
        nx_graph.nodes[nodes]["group"] = 3
        nx_graph.nodes[nodes]["size"] = 20
        if nx_graph.nodes[nodes]["meta"] is not None:
            lsize = str(nx_graph.nodes[nodes]["meta"])
        else:
            lsize = ""
        label = f"{nx_graph.nodes[nodes]['target']}".replace("aten.", "").replace(".default", "") + "\n" + lsize
        nx_graph.nodes[nodes]["label"] = label
    # make the label


# draw networkx graph with directed edges
net = Network(height="100%", width="100%", directed=True,select_menu=True, notebook=True,
             cdn_resources="remote")


# flatten graph to remove nodes that only have 2 input edges where 1 is a const
# make the edges to these nodes go through to the children

illegal_nodes = ["permute", "_unsafe_view", "view", "reshape", "flatten", "expand", "clone", "where", "lift_fresh_copy",
                 "slice", "select", "_to_copy", "getitem"]
has_removed=True
while has_removed:
    has_removed = False
    to_remove = set()
    for node in nx_graph.nodes:
        targets = nx_graph.nodes[node]["target"].split(".")
        print(nx_graph.nodes[node])
        # if any targets are in illegal nodes
        if any([target in illegal_nodes for target in targets]):
            for child in nx_graph.successors(node):
                for parent in nx_graph.predecessors(node):
                    if nx_graph.nodes[parent]["opcode"] == "const":
                        to_remove.add(parent)
                        continue
                    if parent in to_remove:
                        continue
                    nx_graph.add_edge(parent, child)
            to_remove.add(node)
            has_removed = True
    for node in to_remove:
        nx_graph.remove_node(node)

    print("\n\n\n")

# remove const nodes
# to_remove = []
# for node in nx_graph.nodes:
#     if nx_graph.nodes[node]["opcode"] == "const":
#         for child in nx_graph.successors(node):
#             for parent in nx_graph.predecessors(node):
#                 nx_graph.add_edge(parent, child)
#         to_remove.append(node)
# for node in to_remove:
#     nx_graph.remove_node(node)

# change the placeholder node with label "None" to "input" and color it Orange
for node in nx_graph.nodes:
    if nx_graph.nodes[node].get("label") is None:
        continue
    if nx_graph.nodes[node]["label"] == "None":
        nx_graph.nodes[node]["label"] = "input"
        nx_graph.nodes[node]["color"] = "Orange"


net.set_options("""
var options = {
"nodes": {
    "shape": "dot",
    "size": 100,
    "font": {
      "size": 32,
      "align": "center"
    },
    "color": {
      "border": "black",
      "background": "LightBlue",
      "highlight": {
        "border": "black",
        "background": "LightBlue"
      }
    }
},
  "edges": {
    "color": {
      "inherit": false
    },
    "width": 0.5
  },
  "physics": {
    "barnesHut": {
      "gravitationalConstant": -80000,
      "springConstant": 0.5,
      "springLength": 1
    }
  }
}
""")

# render the graph so that there are no overlaps
net.from_nx(nx_graph)

# color the nodes according to their group

# color the
net.show("example.html")
