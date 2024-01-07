import os
import argparse
import io
import tarfile

from tqdm import tqdm
from contextlib import nullcontext
import torch

from crystallm import (
    CIFTokenizer,
    GPT,
    GPTConfig,
)

"""
This script takes a collection of prompts and generates a
corresponding CIF file for each prompt.
"""
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate CIFs from the given prompts.")

    parser.add_argument("--model_dir", type=str, required=True,
                        help="Path to the directory containing the trained model checkpoint file.")
    parser.add_argument("--prompts_file", type=str, required=True,
                        help="Path to the .tar.gz file containing the prompt .txt files.")
    parser.add_argument("--out_file", type=str, required=True,
                        help="Path to the gzipped tarball where the generated CIF files will be stored. "
                             "It is recommended that the filename end in `.tar.gz`.")
    parser.add_argument("--top_k", type=int, default=10,
                        help="The top-k value to use during sampling.")
    parser.add_argument("--max_new_tokens", type=int, default=3000,
                        help="The maximum number of tokens to generate per CIF.")
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"], help="The device to use.")
    parser.add_argument("--temperature", type=float, default=1.0, help="The sampling temperature.")
    parser.add_argument("--seed", type=int, default=1337, help="The random seed.")
    parser.add_argument("--dtype", type=str, default="bfloat16", choices=["float32", "bfloat16", "float16"],
                        help="The datatype to use.")
    parser.add_argument("--num_gens", type=int, default=1,
                        help="The number of times to generate for each CIF.")

    args = parser.parse_args()

    model_dir = args.model_dir
    prompts_file = args.prompts_file
    out_file = args.out_file
    top_k = args.top_k
    max_new_tokens = args.max_new_tokens
    device = args.device
    temperature = args.temperature
    seed = args.seed
    dtype = args.dtype
    num_gens = args.num_gens

    # init torch
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cuda.matmul.allow_tf32 = True  # allow tf32 on matmul
    torch.backends.cudnn.allow_tf32 = True  # allow tf32 on cudnn
    device_type = "cuda" if "cuda" in device else "cpu"  # for later use in torch.autocast
    ptdtype = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}[dtype]
    ctx = nullcontext() if device_type == "cpu" else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

    # init tokenizer
    tokenizer = CIFTokenizer()
    encode = tokenizer.encode
    decode = tokenizer.decode

    print(f"initializing model from {model_dir}...")
    ckpt_path = os.path.join(model_dir, "ckpt.pt")
    checkpoint = torch.load(ckpt_path, map_location=device)
    gptconf = GPTConfig(**checkpoint["model_args"])
    model = GPT(gptconf)
    state_dict = checkpoint["model"]
    unwanted_prefix = "_orig_mod."
    for k, v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
    model.eval()
    model.to(device)
    model = torch.compile(model)  # requires PyTorch 2.0

    prompts = []
    with tarfile.open(prompts_file, "r:gz") as tar:
        for member in tqdm(tar.getmembers(), desc="extracting prompts..."):
            f = tar.extractfile(member)
            if f is not None:
                content = f.read().decode("utf-8")
                cif_id = member.name.replace(".txt", "")
                prompts.append((cif_id, content))

    generated = []
    with torch.no_grad():
        with ctx:
            for id, prompt in tqdm(prompts, desc="generating CIFs from prompts..."):
                start_ids = encode(tokenizer.tokenize_cif(prompt))
                x = torch.tensor(start_ids, dtype=torch.long, device=device)[None, ...]
                gens = []
                for _ in range(num_gens):
                    y = model.generate(x, max_new_tokens, temperature=temperature, top_k=top_k)
                    output = decode(y[0].tolist())
                    gens.append(output)
                generated.append((id, gens))

    with tarfile.open(out_file, "w:gz") as tar:
        for id, gens in tqdm(generated, desc=f"writing CIF files to {out_file}..."):
            for i, cif in enumerate(gens):
                cif_file = tarfile.TarInfo(name=f"{id}__{i+1}.cif")
                cif_bytes = cif.encode("utf-8")
                cif_file.size = len(cif_bytes)
                tar.addfile(cif_file, io.BytesIO(cif_bytes))
