import sys
sys.path.append(".")
import json
import torch
from contextlib import nullcontext
import logging

from nanoGPT.model import GPTConfig, GPT

from lib import get_cif_tokenizer

logger = logging.getLogger(__name__)


def preprocess(comp):
    # NOTE: comp should be a non-reduced formula, like "Na1Cl1"
    logger.info(f"comp received: {comp}")
    return f"data_{comp}\n"


def inference(data, tokenizer, model, device, ctx, num_samples, max_new_tokens, temperature, top_k,
              symmetrized, includes_props):
    encode = tokenizer.encode
    decode = tokenizer.decode

    start_ids = encode(tokenizer.tokenize_cif(data))
    x = (torch.tensor(start_ids, dtype=torch.long, device=device)[None, ...])

    inference_output = []
    with torch.no_grad():
        with ctx:
            for k in range(num_samples):
                y = model.generate(x, max_new_tokens, temperature=temperature, top_k=top_k,
                                   symmetrized=symmetrized, includes_props=includes_props)
                generated_content = decode(y[0].tolist())
                inference_output.append({
                    "input": data,
                    "generated": generated_content,
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

    tokenizer = get_cif_tokenizer(symmetrized=symmetrized, includes_props=includes_props)

    ctx = nullcontext() if device == "cpu" else torch.amp.autocast(device_type=device, dtype=torch.float32)

    data = preprocess(inputs["comp"])

    results = inference(data, tokenizer, model, device, ctx, num_samples, max_new_tokens,
                        temperature, top_k, symmetrized, includes_props)
    return {"cifs": results}


if __name__ == '__main__':
    _model, _config = load_model()
    resp = handle_request(comp="Na1Cl1", context=(_model, _config))
    print(resp)
