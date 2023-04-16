import sys

from torch.utils.data import TensorDataset, DataLoader

sys.path.append(".")
import os
import argparse
import gzip
from tqdm import tqdm

from contextlib import nullcontext
import torch
from nanoGPT.model import GPTConfig, GPT

from lib import get_cif_tokenizer
try:
    import cPickle as pickle
except ImportError:
    import pickle

import warnings
warnings.filterwarnings("ignore")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Read in arguments for your script")

    parser.add_argument('--model_dir', type=str, required=True, help='Model directory')
    parser.add_argument('--eval_file', type=str, required=True, help='evalcifs.pkl.gz file')
    parser.add_argument('--out_file', type=str, required=True, help='output file location')
    parser.add_argument('--symmetrized', action='store_true', default=False, help='Symmetrized flag')
    parser.add_argument('--includes_props', action='store_true', default=False, help='Props flag')
    parser.add_argument('--top_k', type=int, default=10, help='Top K value')
    parser.add_argument('--max_new_tokens', type=int, default=3000, help='Maximum new tokens')
    parser.add_argument('--batch_size', type=int, default=256, help='batch size')
    parser.add_argument('--device', type=str, default='cuda', choices=['cuda', 'cpu'], help='Device to use')

    args = parser.parse_args()

    model_dir = args.model_dir
    eval_fname = args.eval_file
    out_file = args.out_file
    symmetrized = args.symmetrized
    includes_props = args.includes_props
    top_k = args.top_k
    max_new_tokens = args.max_new_tokens
    batch_size = args.batch_size
    device = args.device

    # -----------------------------------------------------------------------------
    temperature = 1.0  # 1.0 = no change, < 1.0 = less random, > 1.0 = more random, in predictions
    seed = 1337
    dtype = 'float16'  # 'float32' or 'bfloat16' or 'float16'
    # -----------------------------------------------------------------------------

    # init torch
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cuda.matmul.allow_tf32 = True  # allow tf32 on matmul
    torch.backends.cudnn.allow_tf32 = True  # allow tf32 on cudnn
    device_type = 'cuda' if 'cuda' in device else 'cpu'  # for later use in torch.autocast
    ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
    ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

    # init tokenizer
    tokenizer = get_cif_tokenizer(symmetrized=symmetrized, includes_props=includes_props)
    encode = tokenizer.encode
    decode = tokenizer.decode

    # init from a model saved in a specific directory
    ckpt_path = os.path.join(model_dir, 'ckpt.pt')
    checkpoint = torch.load(ckpt_path, map_location=device)
    gptconf = GPTConfig(**checkpoint['model_args'])
    model = GPT(gptconf)
    state_dict = checkpoint['model']
    unwanted_prefix = '_orig_mod.'
    for k, v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
    model.eval()
    model.to(device)
    model = torch.compile(model)  # requires PyTorch 2.0

    with gzip.open(eval_fname, "rb") as f:
        eval_cifs = pickle.load(f)

    """
    TODO
    - we can group the prompt ids by sequence length, and then invoke the model on
    batches of these groups, to ensure that all inputs have the same length, and end with a single '\n'
    - we have to just generate until the max length (so we need another model.generate() that doesn't
    stop after double newlines, like model.generate_batch() or something)
    - to fish out the generated CIFs, we have to take all the tokens up to the first occurrence of a
    double newline in each generated CIF string
    """

    grouped = {}
    for eval_cif in tqdm(eval_cifs):
        # append e.g. encoded "data_Na1Cl1\n"
        prompt = eval_cif.split("\n")[0] + "\n"
        start_ids = encode(tokenizer.tokenize_cif(prompt))
        group_size = len(start_ids)
        if group_size not in grouped:
            grouped[group_size] = []
        grouped[group_size].append(start_ids)

    X_groups = {}
    for group, ids in grouped.items():
        x = torch.tensor(ids, dtype=torch.long).pin_memory().to(device, non_blocking=True)
        X_groups[group] = TensorDataset(x)

    generated = []

    pbar = tqdm(total=len(eval_cifs))

    # run generation
    with torch.no_grad():
        with ctx:
            for X in X_groups.values():
                dataloader = DataLoader(X, batch_size=batch_size)
                for batch in dataloader:
                    y = model.generate_batch(batch[0], max_new_tokens, temperature=temperature, top_k=top_k)
                    gen = y.tolist()
                    for g in gen:
                        output = decode(g)
                        generated.append(output)
                    pbar.update(len(batch[0]))

    with gzip.open(out_file, "wb") as f:
        pickle.dump(generated, f)
