import json
import torch
from contextlib import nullcontext
import traceback
from pymatgen.core import Composition

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

    def _postprocess(self, cif_str):
        # replace the symmetry operators with the correct operators
        space_group_symbol = extract_space_group_symbol(cif_str)
        if space_group_symbol is not None and space_group_symbol != "P 1":
            cif_str = replace_symmetry_operators(cif_str, space_group_symbol)

        # remove atom props
        cif_str = remove_atom_props_block(cif_str)

        return cif_str

    def _validate(self, generated_cif, bond_length_acceptability_cutoff):
        if not is_formula_consistent(generated_cif):
            msg = "The generated CIF is inconsistent in terms of composition"
            print(msg)
            return False, msg

        if not is_atom_site_multiplicity_consistent(generated_cif):
            msg = "The generated CIF is inconsistent in terms of atom site multiplicity"
            print(msg)
            return False, msg

        bond_length_score = bond_length_reasonableness_score(generated_cif)
        if bond_length_score < bond_length_acceptability_cutoff:
            msg = f"The bond length score is unacceptable: {bond_length_score:.3f}"
            print(msg)
            return False, msg

        if not is_space_group_consistent(generated_cif):
            msg = "The generated CIF is inconsistent in terms of space group"
            print(msg)
            return False, msg

        return True, None

    def _generate(self, prompt, tokenizer, model, device, ctx, max_new_tokens, temperature, top_k,
                  symmetrized, includes_props, bond_length_acceptability_cutoff):
        print(f"generating from prompt: {prompt}")
        start_ids = tokenizer.encode(tokenizer.tokenize_cif(prompt))
        x = (torch.tensor(start_ids, dtype=torch.long, device=device)[None, ...])
        generated_content = ""
        with torch.no_grad():
            with ctx:
                try:
                    y = model.generate(x, max_new_tokens, temperature=temperature, top_k=top_k,
                                       symmetrized=symmetrized, includes_props=includes_props)
                    generated_content = tokenizer.decode(y[0].tolist())

                    # replace symmetry operators, remove atom props
                    generated_content = self._postprocess(generated_content)

                    valid, msg = self._validate(generated_content, bond_length_acceptability_cutoff)

                except Exception as e:
                    valid = False
                    msg = f"there was an error generating: {e}"
                    print(msg)
                    print(traceback.format_exc())

                return generated_content, valid, msg

    def _get_prompt(self, comp, sg=None):
        # NOTE: we have to use comp.formula, so that the elements are sorted by electronegativity,
        #  which is what the model saw in training; comp.formula looks something like 'Zn1 Cu1 Te1 Se1',
        #  so we have to strip the spaces
        comp_str = comp.formula.replace(" ", "")
        if sg is not None:
            # construct an input string with the space group
            block = get_atomic_props_block_for_formula(comp_str)
            return f"data_{comp_str}\n{block}\n_symmetry_space_group_name_H-M {sg}\n"
        else:
            return f"data_{comp_str}\n"

    def _inference(self, inputs, tokenizer, model, device, ctx, max_new_tokens, temperature, top_k,
                   symmetrized, includes_props, bond_length_acceptability_cutoff):
        reduced_comp = Composition(inputs["comp"])  # TODO try-catch?
        supported_Z = [1, 2, 3, 4, 6, 8]
        sg = inputs["sg"] if "sg" in inputs else None

        if "z" in inputs:
            Z = inputs["z"]
            print(f"Z provided: {Z}")
            comp = Z * reduced_comp
            prompt = self._get_prompt(comp, sg=sg)
            return self._generate(prompt, tokenizer, model, device, ctx,
                                  max_new_tokens, temperature, top_k, symmetrized,
                                  includes_props, bond_length_acceptability_cutoff)
        else:
            #  scan over all z*reduced_comp, return first valid
            print(f"Z not provided, trying Z from {supported_Z}...")
            generated_content = ""
            valid = False
            msg = None
            for Z in supported_Z:
                comp = Z * reduced_comp
                prompt = self._get_prompt(comp, sg=sg)
                generated_content, valid, msg = self._generate(prompt, tokenizer, model, device, ctx,
                                                               max_new_tokens, temperature, top_k, symmetrized,
                                                               includes_props, bond_length_acceptability_cutoff)
                if valid:
                    print(f"The generated CIF is valid for Z={Z}, ending search")
                    break
            return generated_content, valid, msg

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

        inference_output = []
        for _ in range(num_samples):
            generated, valid, msg = self._inference(inputs, tokenizer, self.model, device, ctx,
                                                    max_new_tokens, temperature, top_k, symmetrized,
                                                    includes_props, bond_length_acceptability_cutoff)
            inference_output.append({
                "input": inputs,
                "generated": generated,
                "valid": valid,
                "messages": [msg],
            })

        result = {"cifs": inference_output}
        print(f"returning result: {result}")
        return result

    def __exit__(self, exc_type, exc_val, exc_tb):
        print("exiting CrystaLLM...")


@stub.local_entrypoint()
def main():
    # inp = {"comp": "Na1Cl1"}
    # inp = {"comp": "Na1Cl1", "z": 3}
    # inp = {"comp": "Na1Cl1", "z": 3, "sg": "Pm-3m"}
    # inp = {"comp": "Na1Cl1", "sg": "R-3m"}
    inp = {"comp": "CuSeTeZn"}

    model = CrystaLLMModel()
    model.generate.call(inputs=inp)
