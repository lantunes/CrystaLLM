import sys
sys.path.append(".")
import os
import argparse
import csv
from lib import CIFData, populate_cif_data
from sklearn.metrics import mean_absolute_error, r2_score
from tqdm import tqdm

import pickle
from contextlib import nullcontext
import torch
from nanoGPT.model import GPTConfig, GPT

from lib import get_cif_tokenizer

THIS_DIR = os.path.dirname(os.path.abspath(__file__))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Read in arguments for your script")

    parser.add_argument('--model_dir', type=str, required=True, help='Model directory')
    parser.add_argument('--eval_file', type=str, required=True, help='eval.csv file')
    parser.add_argument('--symmetrized', action='store_true', default=False, help='Symmetrized flag')
    parser.add_argument('--top_k', type=int, default=10, help='Top K value')
    parser.add_argument('--max_new_tokens', type=int, default=500, help='Maximum new tokens')
    parser.add_argument('--device', type=str, default='cuda', choices=['cuda', 'cpu'], help='Device to use')

    args = parser.parse_args()

    model_dir = args.model_dir
    eval_fname = args.eval_file
    symmetrized = args.symmetrized
    top_k = args.top_k
    max_new_tokens = args.max_new_tokens
    device = args.device

    # -----------------------------------------------------------------------------
    temperature = 1.0  # 1.0 = no change, < 1.0 = less random, > 1.0 = more random, in predictions
    seed = 1337
    dtype = 'bfloat16'  # 'float32' or 'bfloat16' or 'float16'
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
    tokenizer = get_cif_tokenizer(symmetrized=symmetrized)
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

    # load the meta.pkl
    meta_path = os.path.join('data', checkpoint['config']['dataset'], 'meta.pkl')
    load_meta = os.path.exists(meta_path)
    if not load_meta:
        raise Exception(f"{meta_path} does not exist")
    print(f"Loading meta from {meta_path}...")
    with open(meta_path, 'rb') as f:
        meta = pickle.load(f)

    n_evaluations = 0
    n_failures = 0

    cell_length_a = {"true": [], "predicted": []}
    cell_length_b = {"true": [], "predicted": []}
    cell_length_c = {"true": [], "predicted": []}
    cell_angle_alpha = {"true": [], "predicted": []}
    cell_angle_beta = {"true": [], "predicted": []}
    cell_angle_gamma = {"true": [], "predicted": []}
    cell_volume = {"true": [], "predicted": []}

    with open(eval_fname, "rt") as f:
        reader = csv.reader(f)
        next(reader)  # skip header
        for line in tqdm(reader):
            cif_data = CIFData.from_csv_row(line)

            start = f"data_{cif_data.composition}\n"

            start_ids = encode(tokenizer.tokenize_cif(start))
            x = (torch.tensor(start_ids, dtype=torch.long, device=device)[None, ...])
            # run generation
            with torch.no_grad():
                with ctx:
                    y = model.generate(x, max_new_tokens, temperature=temperature, top_k=top_k, symmetrized=symmetrized)
                    output = decode(y[0].tolist())

            n_evaluations += 1

            predicted_data = CIFData()
            try:
                # parse the output
                populate_cif_data(predicted_data, output, validate=True)
            except Exception:
                n_failures += 1
                continue

            cell_length_a["true"].append(cif_data.cell_length_a)
            cell_length_a["predicted"].append(predicted_data.cell_length_a)

            cell_length_b["true"].append(cif_data.cell_length_b)
            cell_length_b["predicted"].append(predicted_data.cell_length_b)

            cell_length_c["true"].append(cif_data.cell_length_c)
            cell_length_c["predicted"].append(predicted_data.cell_length_c)

            cell_angle_alpha["true"].append(cif_data.cell_angle_alpha)
            cell_angle_alpha["predicted"].append(predicted_data.cell_angle_alpha)

            cell_angle_beta["true"].append(cif_data.cell_angle_beta)
            cell_angle_beta["predicted"].append(predicted_data.cell_angle_beta)

            cell_angle_gamma["true"].append(cif_data.cell_angle_gamma)
            cell_angle_gamma["predicted"].append(predicted_data.cell_angle_gamma)

            cell_volume["true"].append(cif_data.cell_volume)
            cell_volume["predicted"].append(predicted_data.cell_volume)

    print(f"model_dir: {model_dir}")
    print(f"eval_file: {eval_fname}")
    print(f"evaluations: {n_evaluations:,}, failures: {n_failures:,}")

    print(f"cell_length_a: "
          f"MAE: {mean_absolute_error(cell_length_a['true'], cell_length_a['predicted']):.4f}, "
          f"R2: {r2_score(cell_length_a['true'], cell_length_a['predicted']):.2f}")

    print(f"cell_length_b: "
          f"MAE: {mean_absolute_error(cell_length_b['true'], cell_length_b['predicted']):.4f}, "
          f"R2: {r2_score(cell_length_b['true'], cell_length_b['predicted']):.2f}")

    print(f"cell_length_c: "
          f"MAE: {mean_absolute_error(cell_length_c['true'], cell_length_c['predicted']):.4f}, "
          f"R2: {r2_score(cell_length_c['true'], cell_length_c['predicted']):.2f}")

    print(f"cell_angle_alpha: "
          f"MAE: {mean_absolute_error(cell_angle_alpha['true'], cell_angle_alpha['predicted']):.4f}, "
          f"R2: {r2_score(cell_angle_alpha['true'], cell_angle_alpha['predicted']):.2f}")

    print(f"cell_angle_beta: "
          f"MAE: {mean_absolute_error(cell_angle_beta['true'], cell_angle_beta['predicted']):.4f}, "
          f"R2: {r2_score(cell_angle_beta['true'], cell_angle_beta['predicted']):.2f}")

    print(f"cell_angle_gamma: "
          f"MAE: {mean_absolute_error(cell_angle_gamma['true'], cell_angle_gamma['predicted']):.4f}, "
          f"R2: {r2_score(cell_angle_gamma['true'], cell_angle_gamma['predicted']):.2f}")

    print(f"cell_volume: "
          f"MAE: {mean_absolute_error(cell_volume['true'], cell_volume['predicted']):.4f}, "
          f"R2: {r2_score(cell_volume['true'], cell_volume['predicted']):.2f}")
