import os
import argparse
import torch
import pandas as pd
import csv
from zipfile import ZipFile
from crystallm import (
    GPT,
    GPTConfig,
)
from mcts_sampler import (
    MCTSSampler,
    MCTSEvaluator,
    MCTSLanguageModel,
    ContextSensitiveTreeBuilder,
    PUCTSelector,
    GreedySelector,
    UCTSelector,
)

from crystallm import CIFTokenizer, ZMQScorer


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
        if components[0] == "__MACOSX":
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


def get_formulas_to_process(challenge_set, formulas_to_process):
    formulas = challenge_set.keys()
    if formulas_to_process is not None and formulas_to_process != "":
        print(f"reading formulas to process from {formulas_to_process} ...")
        with open(formulas_to_process, "rt") as f:
            formulas = [formula for formula in f.read().split("\n") if formula != ""]
    return formulas


def perform_random_sampling(state, lm, num_gens, evaluator, top_k, max_new_tokens, newline_id):
    for iter_num in range(1, num_gens + 1):
        print(f"performing simulation {iter_num}...")
        rollout_state = lm.rollout(state, top_k, max_new_tokens, newline_id)
        evaluator(rollout_state, iter_num)


def perform_mcts_sampling(
    start,
    model,
    gptconf,
    num_gens,
    evaluator,
    tokenizer,
    top_child_weight_cutoff,
    n_space_groups,
    bypass_only_child,
    selector,
    c,
    top_k,
    max_new_tokens,
    n_rollouts,
    temperature,
    device,
):
    tree_builder = ContextSensitiveTreeBuilder(
        tokenizer=tokenizer,
        top_child_weight_cutoff=top_child_weight_cutoff,
        n_space_groups=n_space_groups,
        bypass_only_child=bypass_only_child,
    )

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
        width=top_k,
        max_depth=max_new_tokens,
        eval_function=evaluator,
        node_selector=node_selector,
        tokenizer=tokenizer,
        temperature=temperature,
        device=device,
        tree_builder=tree_builder,
    )

    sampler.search(start, num_gens, stepwise=False, n_rollouts=n_rollouts)


def update_results(formula, results_csv_path, results):
    if not os.path.exists(results_csv_path):
        print(f"WARNING: no results.csv file found at {results_csv_path}")
        results.append([formula, 0, float("nan"), float("nan")])
    else:
        # read results .csv
        df_results = pd.read_csv(results_csv_path)
        min_E = df_results["score"].min()
        mean_E = df_results["score"].mean()
        pct_valid = (len(df_results) / num_gens) * 100
        results.append([
            formula, pct_valid, mean_E, min_E
        ])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Perform Challenge")

    parser.add_argument('--model_dir', type=str, required=True, help='Model directory')
    parser.add_argument('--challenge_set', type=str, required=True, help='Path to Challenge set .zip file')
    parser.add_argument('--out_dir', type=str, required=True, help='Directory where the results will be written')
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
    parser.add_argument('--formulas', type=str, required=False, default='',
                        help='Path to file with list of formulas in Challenge set to be processed '
                             '(all formulas will be processed if this is not provided)')
    parser.add_argument('--mcts', action='store_true', default=False, help='Perform MCTS instead of random sampling')
    parser.add_argument('--n_space_groups', type=int, required=False, default=0,
                        help='The number of space groups to use when expanding (MCTS only)')
    parser.add_argument('--selector', type=str, required=False, choices=['puct', 'uct', 'greedy'], default='puct',
                        help='The selection algorithm for MCTS (MCTS only)')
    parser.add_argument('--c', type=float, required=False, default=5.,
                        help='The MCTS selector constant: c_puct for PUCT, c for UCT, epsilon for greedy (MCTS only)')
    parser.add_argument('--reward_k', type=float, required=False, default=2.0,
                        help='The reward scaling factor (MCTS only)')
    parser.add_argument('--top_child_weight_cutoff', type=float, required=False, default=0.9999,
                        help='The top child weight cutoff (MCTS only)')
    parser.add_argument('--bypass_only_child', action='store_true', default=False,
                        help='Whether to bypass the only child (MCTS only)')
    parser.add_argument('--n_rollouts', type=int, required=False, default=1,
                        help='The number of rollouts to perform per simulation (MCTS only)')

    args = parser.parse_args()

    model_dir = args.model_dir
    challenge_set_path = args.challenge_set
    out_dir = args.out_dir
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
    formulas_to_process = args.formulas
    perform_mcts = args.mcts
    n_space_groups = args.n_space_groups
    selector = args.selector
    selector_c = args.c
    reward_k = args.reward_k
    top_child_weight_cutoff = args.top_child_weight_cutoff
    bypass_only_child = args.bypass_only_child
    n_rollouts = args.n_rollouts

    if not os.path.exists(out_dir):
        print(f"creating {out_dir} as it does not exist...")
        os.makedirs(out_dir)

    print(f"reading Challenge set from {challenge_set_path} ...")
    challenge_set = read_challenge_set(challenge_set_path)

    tokenizer = CIFTokenizer()
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

    formulas = get_formulas_to_process(challenge_set, formulas_to_process)
    print(f"formulas to process: {formulas}")

    for formula in formulas:
        print(f"processing {formula} ...")

        if include_space_group:
            start = challenge_set[formula]["prompt_sg"]
        else:
            start = challenge_set[formula]["prompt"]

        formula_dir = os.path.join(out_dir, formula)
        results_csv_path = os.path.join(formula_dir, "results.csv")

        if os.path.exists(formula_dir):
            print(f"directory exists: {formula_dir}; skipping...")
            update_results(formula, results_csv_path, results)
            continue
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

        if perform_mcts:
            perform_mcts_sampling(
                start,
                model,
                gptconf,
                num_gens,
                evaluator,
                tokenizer,
                top_child_weight_cutoff,
                n_space_groups,
                bypass_only_child,
                selector,
                selector_c,
                top_k,
                max_new_tokens,
                n_rollouts,
                temperature,
                device,
            )
        else:
            perform_random_sampling(
                state,
                lm,
                num_gens,
                evaluator,
                top_k,
                max_new_tokens,
                newline_id,
            )

        update_results(formula, results_csv_path, results)

    with open(os.path.join(out_dir, "results.csv"), "wt") as f:
        writer = csv.writer(f)
        writer.writerow(results_header)
        writer.writerows(results)
