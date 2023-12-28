import os
from contextlib import nullcontext
import torch
from crystallm import (
    GPT,
    GPTConfig,
)
from mcts_sampler import MCTSLanguageModel, MCTSEvaluator

from crystallm import CIFTokenizer, ZMQScorer

THIS_DIR = os.path.dirname(os.path.abspath(__file__))

# -----------------------------------------------------------------------------
out_dir = 'out' # ignored if init_from is not 'resume'
temperature = 1.0 # 1.0 = no change, < 1.0 = less random, > 1.0 = more random, in predictions
start = "\n" # or "<|endoftext|>" or etc. Can also specify a file, use as: "FILE:prompt.txt"
seed = 1337
device = 'cuda' # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1', etc.
dtype = 'bfloat16' # 'float32' or 'bfloat16' or 'float16'
compile = False # use PyTorch 2.0 to compile the model to be faster
num_simulations = 200  # the number of simulations to perform
bond_length_acceptability_cutoff = 1.0
top_k = 10
max_new_tokens = 2000
eval_out_dir = 'random'
use_zmq_scorer = True  # must be True, for now
zmq_port = 5555
exec(open(os.path.join(THIS_DIR, 'configurator.py')).read()) # overrides from command line or config file
# -----------------------------------------------------------------------------

torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
device_type = 'cuda' if 'cuda' in device else 'cpu' # for later use in torch.autocast
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

tokenizer = CIFTokenizer()
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
    reward_k=1.0,  # has no meaning for random sampling
    out_dir=eval_out_dir,
)

lm = MCTSLanguageModel(
    model=model,
    config=gptconf,
    child_ids=list(range(len(tokenizer.token_to_id))),
    temperature=temperature,
    device=device,
)

state = tokenizer.encode(tokenizer.tokenize_cif(start))
newline_id = tokenizer.token_to_id["\n"]

for iter_num in range(1, num_simulations + 1):
    print(f"performing simulation {iter_num}...")
    rollout_state = lm.rollout(state, top_k, max_new_tokens, newline_id)
    evaluator(rollout_state, iter_num)
