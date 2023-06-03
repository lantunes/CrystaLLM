import json
import torch
from contextlib import nullcontext
import traceback

from nanoGPT.model import GPTConfig, GPT

from lib import get_cif_tokenizer, bond_length_reasonableness_score, is_formula_consistent, is_space_group_consistent, \
    is_atom_site_multiplicity_consistent, get_atomic_props_block_for_formula, extract_space_group_symbol, \
    replace_symmetry_operators, remove_atom_props_block

import modal
from modal import Image, Stub, Mount, method

image = (
    Image.debian_slim(python_version="3.9")
    .pip_install(
        "pandas==1.5.3",
        "numpy==1.24.2",
        "torch==2.0.1",
        "scikit-learn==1.2.2",
        "tiktoken==0.3.2",
        "transformers==4.27.3",
        "pymatgen==2023.3.23",
    )
)
stub = Stub(
    name="CrystaLLM",
    image=image,
)


@stub.cls(
    shared_volumes={"/crystallm_volume": modal.SharedVolume.from_name("crystallm-volume")},
    mounts=[
        Mount.from_local_dir("./modal_app_config", remote_path="/root"),
        Mount.from_local_file("./lib/spacegroups.txt", remote_path="/root/lib/spacegroups.txt"),
    ],
    gpu="A10G",
)
class CrystaLLMModel:
    def __enter__(self):
        print("enter called: initializing model...")
        config = self._load_config()

        model_path = f"/crystallm_volume/{config['model']}"
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

        self.model = model
        self.config = config

        print("model initialized")

    def _load_config(self):
        config_path = "./modal_config.json"
        with open(config_path) as config_file:
            config = json.load(config_file)
        print(f"serve config: {config}")
        return config

    def _preprocess(self, inputs):
        comp = inputs["comp"]
        # NOTE: comp should be a non-reduced formula, like "Na1Cl1"
        print(f"comp received: {comp}")

        if "sg" in inputs and inputs["sg"] is not None:
            # construct an input string with the space group
            sg = inputs["sg"]
            block = get_atomic_props_block_for_formula(comp)
            return f"data_{comp}\n{block}\n_symmetry_space_group_name_H-M {sg}\n"
        else:
            return f"data_{comp}\n"

    def _is_valid(self, generated_cif, bond_length_acceptability_cutoff):
        bond_length_score = bond_length_reasonableness_score(generated_cif)
        if bond_length_score < bond_length_acceptability_cutoff:
            print(f"bond length score unacceptable: {bond_length_score}")
            return False
        if not is_formula_consistent(generated_cif):
            print("formula inconsistent")
            return False
        if not is_space_group_consistent(generated_cif):
            print("space group inconsistent")
            return False
        if not is_atom_site_multiplicity_consistent(generated_cif):
            print("atom site multiplicity inconsistent")
            return False
        return True

    def _postprocess(self, cif_str):
        # replace the symmetry operators with the correct operators
        space_group_symbol = extract_space_group_symbol(cif_str)
        if space_group_symbol is not None and space_group_symbol != "P 1":
            cif_str = replace_symmetry_operators(cif_str, space_group_symbol)

        # remove atom props
        cif_str = remove_atom_props_block(cif_str)

        return cif_str

    def _inference(self, data, tokenizer, model, device, ctx, num_samples, max_new_tokens, temperature, top_k,
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

                    # replace symmetry operators, remove atom props
                    generated_content = self._postprocess(generated_content)

                    try:
                        valid = self._is_valid(generated_content, bond_length_acceptability_cutoff)
                    except Exception as e:
                        print(f"there was an error validating: {e}")
                        print(traceback.format_exc())
                        valid = False

                    inference_output.append({
                        "input": data,
                        "generated": generated_content,
                        "valid": valid,
                    })

        return inference_output

    @method()
    def generate(self, inputs):
        print("generating...")
        print(f"inputs received: {inputs}")

        device = self.config["device"]
        num_samples = self.config["num_samples"]
        max_new_tokens = self.config["max_new_tokens"]
        temperature = self.config["temperature"]
        top_k = self.config["top_k"]
        symmetrized = self.config["symmetrized"]
        includes_props = self.config["includes_props"]
        bond_length_acceptability_cutoff = self.config["bond_length_acceptability_cutoff"]

        tokenizer = get_cif_tokenizer(symmetrized=symmetrized, includes_props=includes_props)

        ctx = nullcontext() if device == "cpu" else torch.amp.autocast(device_type=device, dtype=torch.float32)

        data = self._preprocess(inputs)

        results = self._inference(data, tokenizer, self.model, device, ctx, num_samples, max_new_tokens,
                                  temperature, top_k, symmetrized, includes_props, bond_length_acceptability_cutoff)

        result = {"cifs": results}
        print(f"returning result: {result}")
        return result

    def __exit__(self, exc_type, exc_val, exc_tb):
        print("exiting CrystaLLM...")


@stub.local_entrypoint()
def main():
    model = CrystaLLMModel()
    model.generate.call(inputs={"comp": "Na1Cl1"})
