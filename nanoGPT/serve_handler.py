import os
from ts.torch_handler.base_handler import BaseHandler
import logging
import ast
import json
import torch
from contextlib import nullcontext
import traceback
from pymatgen.core import Composition

from model import GPTConfig, GPT

from lib import get_cif_tokenizer, bond_length_reasonableness_score, is_formula_consistent, is_space_group_consistent, \
    is_atom_site_multiplicity_consistent, get_atomic_props_block_for_formula, extract_space_group_symbol, \
    replace_symmetry_operators, remove_atom_props_block

logger = logging.getLogger(__name__)


class NanoGPTHandler(BaseHandler):

    def __init__(self):
        super().__init__()
        self.model = None

    def initialize(self, context):
        properties = context.system_properties
        model_dir = properties.get("model_dir")

        serve_config_path = os.path.join(model_dir, "serve_config.json")
        if os.path.isfile(serve_config_path):
            with open(serve_config_path) as serve_config_file:
                serve_config = json.load(serve_config_file)
            logger.info(f"serve config: {serve_config}")
        else:
            logger.warning("Missing the serve_config.json file.")

        device = serve_config["device"]
        compile = serve_config["compile"]

        checkpoint = torch.load("ckpt.pt", map_location=device)
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
        self.config = serve_config

        logger.info("model initialized")

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
            logger.info(msg)
            return False, msg

        if not is_atom_site_multiplicity_consistent(generated_cif):
            msg = "The generated CIF is inconsistent in terms of atom site multiplicity"
            logger.info(msg)
            return False, msg

        bond_length_score = bond_length_reasonableness_score(generated_cif)
        if bond_length_score < bond_length_acceptability_cutoff:
            msg = f"Unreasonable bond lengths detected " \
                  f"({(1-bond_length_score)*100:.0f}% of bond lengths were found to be unreasonable)"
            logger.info(msg)
            return False, msg

        if not is_space_group_consistent(generated_cif):
            msg = "The generated CIF is inconsistent in terms of space group"
            logger.info(msg)
            return False, msg

        return True, None

    def _generate(self, prompt, tokenizer, model, device, ctx, max_new_tokens, temperature, top_k,
                  symmetrized, includes_props, bond_length_acceptability_cutoff):
        logger.info(f"generating from prompt: {prompt}")
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
                    logger.info(msg)
                    logger.info(traceback.format_exc())

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

    def _get_unsupported_atoms(self, comp, supported_atoms):
        # The model only supports elements up to a certain atomic number
        supported_atoms = set(supported_atoms)
        unsupported_atoms = []
        for e in comp.elements:
            if e.name not in supported_atoms:
                unsupported_atoms.append(e.name)
        return unsupported_atoms

    def _inference(self, inputs, tokenizer, model, device, ctx, max_new_tokens, temperature, top_k,
                   symmetrized, includes_props, bond_length_acceptability_cutoff):
        reduced_comp = Composition(inputs["comp"])  # TODO try-catch?

        unsupported_atoms = self._get_unsupported_atoms(reduced_comp, tokenizer.atoms())
        if len(unsupported_atoms) > 0:
            return "", False, f"composition contains unsupported atoms: {', '.join(unsupported_atoms)}"

        supported_Z = [1, 2, 3, 4, 6, 8]
        sg = inputs["sg"] if "sg" in inputs else None

        if "z" in inputs:
            Z = inputs["z"]
            logger.info(f"Z provided: {Z}")
            comp = Z * reduced_comp
            prompt = self._get_prompt(comp, sg=sg)
            return self._generate(prompt, tokenizer, model, device, ctx,
                                  max_new_tokens, temperature, top_k, symmetrized,
                                  includes_props, bond_length_acceptability_cutoff)
        else:
            #  scan over all z*reduced_comp, return first valid
            logger.info(f"Z not provided, trying Z from {supported_Z}...")
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
                    logger.info(f"The generated CIF is valid for Z={Z}, ending search")
                    break
            return generated_content, valid, msg

    def preprocess(self, data):
        inputs = data[0].get("data") or data[0].get("body")
        inputs = inputs.decode("utf-8")
        return ast.literal_eval(inputs)

    def inference(self, inputs, *args, **kwargs):
        logger.info("generating...")
        logger.info(f"inputs received: {inputs}")

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
        logger.info(f"returning result: {result}")
        return [json.dumps(result)]

    def postprocess(self, data):
        return data
