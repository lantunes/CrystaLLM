import sys
sys.path.append(".")
import os
import argparse
import torch
import pandas as pd
import csv
from model import GPTConfig, GPT
from zipfile import ZipFile
from mcts_sampler import MCTSLanguageModel, MCTSEvaluator

from lib import get_cif_tokenizer, ZMQScorer


def prepare_model(seed, model_dir, device, compile):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cuda.matmul.allow_tf32 = True  # allow tf32 on matmul
    torch.backends.cudnn.allow_tf32 = True  # allow tf32 on cudnn

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
    if compile:
        model = torch.compile(model)  # requires PyTorch 2.0 (optional)

    return model, gptconf


def read_challenge_set(challenge_set_path):
    input_zip = ZipFile(challenge_set_path)

    # map from formula -> {"props": <props file content>, "props_sg": <props sg file content>}
    challenge_set = {}

    for zipfile in input_zip.filelist:
        components = zipfile.filename.split("/")
        if len(components) < 3 or len(components[-1]) == 0:
            continue
        formula = components[1]
        if formula not in challenge_set:
            challenge_set[formula] = {}

        fname = components[2]
        if fname.endswith("props.txt"):
            content = input_zip.read(zipfile.filename).decode("utf-8")
            challenge_set[formula]["prompt"] = content

        if fname.endswith("props_sg.txt"):
            content = input_zip.read(zipfile.filename).decode("utf-8")
            challenge_set[formula]["prompt_sg"] = content

    return challenge_set


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Perform Challenge")

    parser.add_argument('--model_dir', type=str, required=True, help='Model directory')
    parser.add_argument('--challenge_set', type=str, required=True, help='Path to Challenge set .zip file')
    parser.add_argument('--out_dir', type=str, required=True, help='Directory where the results will be written')
    parser.add_argument('--symmetrized', action='store_true', default=False, help='Symmetrized flag')
    parser.add_argument('--includes_props', action='store_true', default=False, help='Props flag')
    parser.add_argument('--top_k', type=int, default=10, help='Top K value')
    parser.add_argument('--temperature', type=float, default=1.0, help='The sampling temperature')
    parser.add_argument('--max_new_tokens', type=int, default=2000, help='Maximum new tokens')
    parser.add_argument('--device', type=str, default='cuda', choices=['cuda', 'cpu'], help='Device to use')
    parser.add_argument('--num_gens', type=int, default=100, help='The number of times to generate for each CIF')
    parser.add_argument('--seed', type=int, default=1337, help='The random seed')
    parser.add_argument('--dtype', type=str, default='float16', choices=['float32', 'bfloat16', 'float16'],
                        help='Datatype')
    parser.add_argument('--use_zmq_scorer', action='store_true', default=True, help='Use ZMQ scorer')
    parser.add_argument('--zmq_port', type=int, default=5555, help='ZMQ port')
    parser.add_argument('--bond_length_acceptability_cutoff', type=float, default=1.0,
                        help='The bond length acceptability cutoff')
    parser.add_argument('--include_space_group', action='store_true', default=False,
                        help='Include the space group in the prompt')
    parser.add_argument('--compile', action='store_true', default=False, help='Compile model')

    args = parser.parse_args()

    model_dir = args.model_dir
    challenge_set_path = args.challenge_set
    out_dir = args.out_dir
    symmetrized = args.symmetrized
    includes_props = args.includes_props
    top_k = args.top_k
    temperature = args.temperature
    max_new_tokens = args.max_new_tokens
    device = args.device
    num_gens = args.num_gens
    seed = args.seed
    dtype = args.dtype
    use_zmq_scorer = args.use_zmq_scorer
    zmq_port = args.zmq_port
    bond_length_acceptability_cutoff = args.bond_length_acceptability_cutoff
    include_space_group = args.include_space_group
    compile = args.compile

    if not os.path.exists(out_dir):
        print(f"creating {out_dir} as it does not exist...")
        os.makedirs(out_dir)

    print(f"reading Challenge set from {challenge_set_path} ...")
    challenge_set = read_challenge_set(challenge_set_path)

    tokenizer = get_cif_tokenizer(symmetrized=symmetrized, includes_props=includes_props)
    encode = tokenizer.encode
    decode = tokenizer.decode

    print(f"reading model from {model_dir} ...")
    model, gptconf = prepare_model(seed, model_dir, device, compile)

    scorer = ZMQScorer(port=zmq_port) if use_zmq_scorer else None

    lm = MCTSLanguageModel(
        model=model,
        config=gptconf,
        child_ids=list(range(len(tokenizer.token_to_id))),
        temperature=temperature,
        device=device,
    )

    results_header = ["formula", "validity_rate", "mean_E", "min_E"]
    results = []

    for formula in challenge_set:
        print(f"processing {formula} ...")

        if include_space_group:
            start = challenge_set[formula]["prompt_sg"]
        else:
            start = challenge_set[formula]["prompt"]

        formula_dir = os.path.join(out_dir, formula)
        os.makedirs(formula_dir)

        evaluator = MCTSEvaluator(
            scorer=scorer,
            tokenizer=tokenizer,
            bond_length_acceptability_cutoff=bond_length_acceptability_cutoff,
            reward_k=1.0,  # has no meaning for random sampling
            out_dir=formula_dir,
        )

        state = tokenizer.encode(tokenizer.tokenize_cif(start))
        newline_id = tokenizer.token_to_id["\n"]

        for iter_num in range(1, num_gens + 1):
            print(f"performing simulation {iter_num}...")
            rollout_state = lm.rollout(state, top_k, max_new_tokens, newline_id)
            evaluator(rollout_state, iter_num)

        # read results .csv
        results_csv_path = os.path.join(formula_dir, "results.csv")
        if not os.path.exists(results_csv_path):
            print(f"WARNING: no results.csv file found at {results_csv_path}")
            results.append([formula, 0, float("nan"), float("nan")])
            continue

        df_results = pd.read_csv(results_csv_path)
        min_E = df_results["score"].min()
        mean_E = df_results["score"].mean()
        pct_valid = (len(df_results) / num_gens) * 100
        results.append([
            formula, pct_valid, mean_E, min_E
        ])

    with open(os.path.join(out_dir, "results.csv"), "wt") as f:
        writer = csv.writer(f)
        writer.writerow(results_header)
        writer.writerows(results)
