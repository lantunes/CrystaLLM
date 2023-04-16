import sys
sys.path.append(".")
import os
import argparse
import torch
import torch.multiprocessing as mp
import gzip

from tqdm import tqdm

from contextlib import nullcontext
from nanoGPT.model import GPTConfig, GPT

from lib import get_cif_tokenizer, array_split
try:
    import cPickle as pickle
except ImportError:
    import pickle


def progress_listener(queue, n):
    pbar = tqdm(total=n)
    while True:
        message = queue.get()
        if message == "kill":
            break
        pbar.update(message)


def generate(ctx, model, chunk_of_prompts, queue, max_new_tokens, temperature, top_k, symmetrized, includes_props):
    model = torch.compile(model)  # requires PyTorch 2.0
    outputs = []
    with torch.no_grad():
        with ctx:
            for x in chunk_of_prompts:
                y = model.generate(x, max_new_tokens, temperature=temperature, top_k=top_k,
                                   symmetrized=symmetrized, includes_props=includes_props)
                outputs.append(y[0].tolist())
                queue.put(1)
    return outputs


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Read in arguments for your script")

    parser.add_argument('--model_dir', type=str, required=True, help='Model directory')
    parser.add_argument('--eval_file', type=str, required=True, help='evalcifs.pkl.gz file')
    parser.add_argument('--out_file', type=str, required=True, help='output file location')
    parser.add_argument('--symmetrized', action='store_true', default=False, help='Symmetrized flag')
    parser.add_argument('--includes_props', action='store_true', default=False, help='Props flag')
    parser.add_argument('--top_k', type=int, default=10, help='Top K value')
    parser.add_argument('--max_new_tokens', type=int, default=3000, help='Maximum new tokens')
    parser.add_argument('--num_workers', type=int, default=2, help='Number of workers')
    parser.add_argument('--device', type=str, default='cuda', choices=['cuda', 'cpu'], help='Device to use')

    args = parser.parse_args()

    model_dir = args.model_dir
    eval_fname = args.eval_file
    out_file = args.out_file
    symmetrized = args.symmetrized
    includes_props = args.includes_props
    top_k = args.top_k
    max_new_tokens = args.max_new_tokens
    device = args.device
    num_workers = args.num_workers

    # -----------------------------------------------------------------------------
    temperature = 1.0  # 1.0 = no change, < 1.0 = less random, > 1.0 = more random, in predictions
    seed = 1337
    dtype = 'float16'  # 'float32' or 'bfloat16' or 'float16'
    # -----------------------------------------------------------------------------

    mp.set_start_method('spawn', force=True)

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

    model.share_memory()

    with gzip.open(eval_fname, "rb") as f:
        eval_cifs = pickle.load(f)

    X = []
    for eval_cif in tqdm(eval_cifs):
        # append e.g. encoded "data_Na1Cl1\n"
        prompt = eval_cif.split("\n")[0] + "\n"
        start_ids = encode(tokenizer.tokenize_cif(prompt))
        X.append((torch.tensor(start_ids, dtype=torch.long, device=device)[None, ...]))

    chunks = array_split(X, num_workers)

    manager = mp.Manager()
    queue = manager.Queue()
    pool = mp.Pool(num_workers)

    watcher = pool.apply_async(progress_listener, (queue, len(X),))

    jobs = []
    for i in range(num_workers):
        chunk = chunks[i]
        job = pool.apply_async(generate, (ctx, model, chunk, queue, max_new_tokens, temperature, top_k, symmetrized, includes_props))
        jobs.append(job)

    generated = []
    for job in jobs:
        generated.extend([decode(o) for o in job.get()])

    queue.put("kill")
    pool.close()
    pool.join()

    with gzip.open(out_file, "wb") as f:
        pickle.dump(generated, f)
