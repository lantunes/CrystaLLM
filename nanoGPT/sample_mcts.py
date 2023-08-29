"""
Sample from a trained model using MCTS
"""
import sys

sys.path.append(".")
import os
from contextlib import nullcontext
import torch
from model import GPTConfig, GPT
from mcts_sampler import (
    MCTSSampler,
    MCTSEvaluator,
    ContextSensitiveTreeBuilder,
    PUCTSelector,
    GreedySelector,
    UCTSelector,
)

from lib import get_cif_tokenizer, ZMQScorer

THIS_DIR = os.path.dirname(os.path.abspath(__file__))

# -----------------------------------------------------------------------------
out_dir = 'out' # ignored if init_from is not 'resume'
temperature = 1.0 # 1.0 = no change, < 1.0 = less random, > 1.0 = more random, in predictions
start = "\n" # or "<|endoftext|>" or etc. Can also specify a file, use as: "FILE:prompt.txt"
seed = 1337
device = 'cuda' # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1', etc.
dtype = 'bfloat16' # 'float32' or 'bfloat16' or 'float16'
compile = False # use PyTorch 2.0 to compile the model to be faster
symmetrized = True # whether the CIF files are symmetrized
includes_props = True # whether CIF files contain an atomic properties section
tree_width = 10  # the tree width
max_depth = 1000  # the maximum depth of the tree
c = 5.  # the selector constant: c_puct for PUCT, c for UCT, epsilon for greedy
num_simulations = 200  # the number of simulations to perform during search
bond_length_acceptability_cutoff = 1.0
reward_k = 2.0
mcts_out_dir = 'mcts'
use_zmq_scorer = True  # must be True, for now
zmq_port = 5555
use_context_sensitive_tree_builder = True
top_child_weight_cutoff = 0.99
stepwise = False
selector = 'puct'  # valid values: 'puct', 'uct', 'greedy'
exec(open(os.path.join(THIS_DIR, 'configurator.py')).read()) # overrides from command line or config file
# -----------------------------------------------------------------------------

torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
device_type = 'cuda' if 'cuda' in device else 'cpu' # for later use in torch.autocast
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

tokenizer = get_cif_tokenizer(symmetrized=symmetrized, includes_props=includes_props)
encode = tokenizer.encode
decode = tokenizer.decode

# init from a model saved in a specific directory
ckpt_path = os.path.join(out_dir, 'ckpt.pt')
checkpoint = torch.load(ckpt_path, map_location=device)
gptconf = GPTConfig(**checkpoint['model_args'])
model = GPT(gptconf)
state_dict = checkpoint['model']
unwanted_prefix = '_orig_mod.'
for k,v in list(state_dict.items()):
    if k.startswith(unwanted_prefix):
        state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
model.load_state_dict(state_dict)

model.eval()
model.to(device)
if compile:
    model = torch.compile(model) # requires PyTorch 2.0 (optional)

# encode the beginning of the prompt
if start.startswith('FILE:'):
    with open(start[5:], 'r', encoding='utf-8') as f:
        start = f.read()

scorer = ZMQScorer(port=zmq_port) if use_zmq_scorer else None

evaluator = MCTSEvaluator(
    scorer=scorer,
    tokenizer=tokenizer,
    bond_length_acceptability_cutoff=bond_length_acceptability_cutoff,
    reward_k=reward_k,
    out_dir=mcts_out_dir,
)

tree_builder = ContextSensitiveTreeBuilder(top_child_weight_cutoff) if use_context_sensitive_tree_builder else None

if selector == "puct":
    node_selector = PUCTSelector(cpuct=c)
elif selector == "greedy":
    node_selector = GreedySelector(epsilon=c)
elif selector == "uct":
    node_selector = UCTSelector(c=c)
else:
    raise Exception(f"unsupported selector: {selector}")

sampler = MCTSSampler(
    model=model,
    config=gptconf,
    width=tree_width,
    max_depth=max_depth,
    eval_function=evaluator,
    node_selector=node_selector,
    tokenizer=tokenizer,
    temperature=temperature,
    device=device,
    tree_builder=tree_builder,
)

if stepwise:
    print("performing stepwise search...")

    while True:
        print(start)
        print()
        # run search and get most visited state
        most_visited_state = sampler.search(start, num_simulations, stepwise=True)
        print(f"most visited token: {repr(tokenizer.decode([most_visited_state[-1]]))}")
        start = tokenizer.decode(most_visited_state)
        if start[-2:] == ["\n\n"]:
            break

    print(start)
    # TODO validate and preprocess cif
    print("invoking external scorer on final sequence...")
    score = scorer.score(start)
    print(f"external scorer returned score: {score}")

else:
    sampler.search(start, num_simulations, stepwise=False)
