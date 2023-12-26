"""
Sample from a trained model
"""
import os
from contextlib import nullcontext
import torch
from crystallm import (
    GPT,
    GPTConfig,
)

from crystallm import get_cif_tokenizer

THIS_DIR = os.path.dirname(os.path.abspath(__file__))

# -----------------------------------------------------------------------------
init_from = 'resume' # either 'resume' (from an out_dir) or a gpt2 variant (e.g. 'gpt2-xl')
out_dir = 'out' # ignored if init_from is not 'resume'
start = "\n" # or "<|endoftext|>" or etc. Can also specify a file, use as: "FILE:prompt.txt"
num_samples = 10 # number of samples to draw
max_new_tokens = 500 # number of tokens generated in each sample
temperature = 0.8 # 1.0 = no change, < 1.0 = less random, > 1.0 = more random, in predictions
top_k = 200 # retain only the top_k most likely tokens, clamp others to have 0 probability
seed = 1337
device = 'cuda' # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1', etc.
dtype = 'bfloat16' # 'float32' or 'bfloat16' or 'float16'
compile = False # use PyTorch 2.0 to compile the model to be faster
exec(open(os.path.join(THIS_DIR, 'configurator.py')).read()) # overrides from command line or config file
# -----------------------------------------------------------------------------

torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
device_type = 'cuda' if 'cuda' in device else 'cpu' # for later use in torch.autocast
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

tokenizer = get_cif_tokenizer(symmetrized=True, includes_props=True)
encode = tokenizer.encode
decode = tokenizer.decode

# model
if init_from == 'resume':
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

if start.startswith('FILE:'):
    with open(start[5:], 'r', encoding='utf-8') as f:
        lines = f.read().splitlines()  # split the file content into lines

    # encode each line separately and store the results in a list
    encoded_lines = [torch.tensor(encode(tokenizer.tokenize_cif(line)), dtype=torch.long, device=device)[None, ...] for line in lines]

    # concatenate the encoded lines along the batch dimension to create a single tensor
    x = torch.cat(encoded_lines, dim=0)
else:
    start_ids = encode(tokenizer.tokenize_cif(start))
    x = torch.tensor(start_ids, dtype=torch.long, device=device)[None, ...]


# run generation
with torch.no_grad():
    with ctx:
        for k in range(num_samples):
            y = model.generate_batch(x, max_new_tokens, temperature=temperature, top_k=top_k)
            for j in range(len(y)):
                print(decode(y[j].tolist()))
                print('- - - - - - - - - - -')
            print('==============================')
