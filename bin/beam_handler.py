import sys
sys.path.append(".")
import json
import torch
from contextlib import nullcontext
import logging

from nanoGPT.model import GPTConfig, GPT

from lib import get_cif_tokenizer, bond_length_reasonableness_score, is_formula_consistent, is_space_group_consistent, \
    is_atom_site_multiplicity_consistent, get_atomic_props_block_for_formula

logger = logging.getLogger(__name__)


def preprocess(inputs):
    comp = inputs["comp"]
    # NOTE: comp should be a non-reduced formula, like "Na1Cl1"
    logger.info(f"comp received: {comp}")

    if "sg" in inputs and inputs["sg"] is not None:
        # construct an input string with the space group
        sg = inputs["sg"]
        block = get_atomic_props_block_for_formula(comp)
        return f"data_{comp}\n{block}\n_symmetry_space_group_name_H-M {sg}\n"
    else:
        return f"data_{comp}\n"


def is_valid(generated_cif, bond_length_acceptability_cutoff):
    bond_length_score = bond_length_reasonableness_score(generated_cif)
    if bond_length_score < bond_length_acceptability_cutoff:
        logger.info(f"bond length score unacceptable: {bond_length_score}")
        return False
    if not is_formula_consistent(generated_cif):
        logger.info("formula inconsistent")
        return False
    if not is_space_group_consistent(generated_cif):
        logger.info("space group inconsistent")
        return False
    if not is_atom_site_multiplicity_consistent(generated_cif):
        logger.info("atom site multiplicity inconsistent")
        return False
    return True


def inference(data, tokenizer, model, device, ctx, num_samples, max_new_tokens, temperature, top_k,
              symmetrized, includes_props, bond_length_acceptability_cutoff):

    start_ids = tokenizer.encode(tokenizer.tokenize_cif(data))
    x = (torch.tensor(start_ids, dtype=torch.long, device=device)[None, ...])

    inference_output = []
    with torch.no_grad():
        with ctx:
            for k in range(num_samples):
                y = model.generate(x, max_new_tokens, temperature=temperature, top_k=top_k,
                                   symmetrized=symmetrized, includes_props=includes_props)
                generated_content = tokenizer.decode(y[0].tolist())

                try:
                    valid = is_valid(generated_content, bond_length_acceptability_cutoff)
                except Exception as e:
                    logger.exception(f"there was an error validating: {e}")
                    valid = False

                inference_output.append({
                    "input": data,
                    "generated": generated_content,
                    "valid": valid,
                })

    return inference_output


def load_config():
    config_path = "./beam_config.json"
    with open(config_path) as config_file:
        config = json.load(config_file)
    logger.info(f"serve config: {config}")
    return config


def load_model():
    config = load_config()

    model_path = f"./saved_models/{config['model']}"
    device = config["device"]
    compile = config["compile"]

    checkpoint = torch.load(model_path, map_location=device)
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
    if compile:
        model = torch.compile(model)

    return model, config


def handle_request(**inputs):
    model, config = inputs["context"]

    device = config["device"]
    num_samples = config["num_samples"]
    max_new_tokens = config["max_new_tokens"]
    temperature = config["temperature"]
    top_k = config["top_k"]
    symmetrized = config["symmetrized"]
    includes_props = config["includes_props"]
    bond_length_acceptability_cutoff = config["bond_length_acceptability_cutoff"]

    tokenizer = get_cif_tokenizer(symmetrized=symmetrized, includes_props=includes_props)

    ctx = nullcontext() if device == "cpu" else torch.amp.autocast(device_type=device, dtype=torch.float32)

    data = preprocess(inputs)

    results = inference(data, tokenizer, model, device, ctx, num_samples, max_new_tokens,
                        temperature, top_k, symmetrized, includes_props, bond_length_acceptability_cutoff)

    return {"cifs": results}


if __name__ == '__main__':
    _model, _config = load_model()
    resp = handle_request(comp="Na1Cl1", context=(_model, _config))
    print(resp)
