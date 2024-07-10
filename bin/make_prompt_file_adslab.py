import os, sys
import argparse
import pandas as pd
from tqdm import tqdm
import tarfile
import io

# Calculate the absolute path to the 'crystallm' directory
script_dir = os.path.dirname(os.path.abspath(__file__))  # Absolute dir the script is in
crystallm_dir = os.path.abspath(os.path.join(script_dir, '..'))  # 'crystallm' is one level up

# Add 'crystallm' directory to sys.path
sys.path.append(crystallm_dir)

from crystallm import get_atomic_props_block_for_formula, get_string_from_symbols


def get_prompt(comp, sg=None):
    # NOTE: we have to use comp.formula, so that the elements are sorted by electronegativity,
    #  which is what the model saw in training; comp.formula looks something like 'Zn1 Cu1 Te1 Se1',
    #  so we have to strip the spaces
    comp_str = comp.formula.replace(" ", "")
    # change miller index to string. ex. (1, 1, 1) to 111
    # miller_str = ''.join(str(index) for index in miller) if isinstance(miller, (list, tuple)) else str(miller)
    if sg is not None:
        # construct an input string with the space group
        block = get_atomic_props_block_for_formula(comp_str)
        return f"data_{comp_str}\n{block}\n_symmetry_space_group_name_H-M {sg}\n"
    else:
        return f"data_{comp_str}\n"


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Construct prompt files from the given directory contraining CIF files.")
    # parser.add_argument("cif_dir", type=str, help="The directory containing CIF files.")
    parser.add_argument("sid_list_file", type=str, help="The file containing the list of system ids.")
    parser.add_argument("prompt_fname", type=str,
                        help="The path to the pickle file where the prompt will be stored.")
    parser.add_argument("--meta_path", type=str, default="/home/jovyan/CATBERT/metadata/oc20_meta/oc20_data_metadata.pkl",
                        help="The path to the metadata file (optional).")
    args = parser.parse_args()
    sid_list_file = args.sid_list_file
    prompt_fname = args.prompt_fname
    meta_path = args.meta_path

    metadata = pd.read_pickle(meta_path)
    print('metadata loaded')
    initial_prompts = []
    sid_list = pd.read_pickle(sid_list_file)
    print('sid_list loaded')

    with tarfile.open(prompt_fname, "w:gz") as tar:
        for sid in tqdm(sid_list, desc="preparing prompts..."):
            bulk_symbols = metadata[sid]["bulk_symbols"]
            ads_symbols = metadata[sid]["ads_symbols"]
            bulk_str, ads_str = get_string_from_symbols(bulk_symbols, ads_symbols)
            # comp = '-'.join([bulk_str, ads_str])
            prompt = f"data_{bulk_str}-{ads_str}\n"
            # prompt = get_prompt(comp)
            prompt_file = tarfile.TarInfo(name=f"{sid}.txt")
            prompt_bytes = prompt.encode("utf-8")
            prompt_file.size = len(prompt_bytes)
            breakpoint()
            tar.addfile(prompt_file, io.BytesIO(prompt_bytes))

    # for sid in sid_list:
    #     bulk_symbols = metadata[sid]["bulk_symbols"]
    #     ads_symbols = metadata[sid]["ads_symbols"]
    #     bulk_str, ads_str = get_string_from_symbols(bulk_symbols, ads_symbols)
    #     comp = '-'.join([bulk_str, ads_str])
    #     prompt = get_prompt(comp)
    #     initial_prompts.append(prompt)

    # print(f"writing prompt to {prompt_fname} ...")
    # with open(prompt_fname, "wb") as f:
    #     pickle.dump(initial_prompts, f)

    # with tarfile.open(out_fname, "w:gz") as tar:
    #     for id, cif in tqdm(cifs, desc="preparing prompts..."):
    #         prompt = extract_prompt(cif, PATTERN_COMP_SG if with_spacegroup else PATTERN_COMP)

    #         prompt_file = tarfile.TarInfo(name=f"{id}.txt")
    #         prompt_bytes = prompt.encode("utf-8")
    #         prompt_file.size = len(prompt_bytes)
    #         tar.addfile(prompt_file, io.BytesIO(prompt_bytes))