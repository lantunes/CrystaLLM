import os, sys
import argparse
import glob
import pickle
import pandas as pd
import tqdm
from pymatgen.core import Composition

# Calculate the absolute path to the 'crystallm' directory
script_dir = os.path.dirname(os.path.abspath(__file__))  # Absolute dir the script is in
crystallm_dir = os.path.abspath(os.path.join(script_dir, '..'))  # 'crystallm' is one level up

# Add 'crystallm' directory to sys.path
sys.path.append(crystallm_dir)

from crystallm import get_atomic_props_block_for_formula


def get_prompt(comp, miller, sg=None):
    # NOTE: we have to use comp.formula, so that the elements are sorted by electronegativity,
    #  which is what the model saw in training; comp.formula looks something like 'Zn1 Cu1 Te1 Se1',
    #  so we have to strip the spaces
    comp_str = comp.formula.replace(" ", "")
    # change miller index to string. ex. (1, 1, 1) to 111
    miller_str = ''.join(str(index) for index in miller) if isinstance(miller, (list, tuple)) else str(miller)
    if sg is not None:
        # construct an input string with the space group
        block = get_atomic_props_block_for_formula(comp_str)
        return f"data_{comp_str}\n{block}\n_symmetry_space_group_name_H-M {sg}\n"
    else:
        return f"data_{comp_str}_miller_{miller_str}\n"


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Construct prompt files from the given directory contraining CIF files.")
    parser.add_argument("cif_dir", type=str, help="The directory containing CIF files.")
    parser.add_argument("prompt_fname", type=str,
                        help="The path to the pickle file where the prompt will be stored.")
    parser.add_argument("--meta_path", type=str, default="/home/jovyan/CATBERT/metadata/oc20_meta/oc20_data_metadata.pkl",
                        help="The path to the metadata file (optional).")
    args = parser.parse_args()
    cif_dir = args.cif_dir
    prompt_fname = args.prompt_fname
    meta_path = args.meta_path

    metadata = pd.read_pickle(meta_path)
    # obtain "_chemical_formula_structural <composition>" block from CIF files
    cif_files = glob.glob(os.path.join(cif_dir, "*.cif"))
    initial_prompts = []
    for cif_file in tqdm.tqdm(cif_files):
        sid = os.path.splitext(os.path.basename(cif_file))[0]
        miller = metadata[sid]["miller_index"]
        with open(cif_file, "rt") as f:
            for line in f:
                if line.startswith("_chemical_formula_structural"):
                    comp_str = line.split()[1]
                    comp = Composition(comp_str)
                    prompt = get_prompt(comp, miller)
                    initial_prompts.append(prompt)
                    break

    print(f"writing prompt to {prompt_fname} ...")
    with open(prompt_fname, "wb") as f:
        pickle.dump(initial_prompts, f)
