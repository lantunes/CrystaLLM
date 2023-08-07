import gzip
import pandas as pd
import re
from tqdm import tqdm
from pymatgen.core import Structure
from pymatgen.analysis.structure_matcher import StructureMatcher, ElementComparator

from lib import extract_data_formula, extract_space_group_symbol, replace_symmetry_operators, remove_atom_props_block, \
    extract_numeric_property, get_unit_cell_volume

try:
    import cPickle as pickle
except ImportError:
    import pickle

import warnings
warnings.filterwarnings("ignore")


def clean_cif(cif_str):
    cif_str = re.sub(' +', ' ', cif_str)
    cif_str = re.sub('\n +', '\n', cif_str)

    # replace the symmetry operators with the correct operators
    space_group_symbol = extract_space_group_symbol(cif_str)
    if space_group_symbol is not None and space_group_symbol != "P 1":
        cif_str = replace_symmetry_operators(cif_str, space_group_symbol)

    # remove atom props
    cif_str = remove_atom_props_block(cif_str)

    return cif_str


if __name__ == '__main__':

    true_cifs_fname = "../out/orig_cifs_mp_2022_04_12+oqmd_v1_5+nomad_2023_04_30__comp-sg_augm.test.pkl.gz"
    # this should be a list of lists of k generation attempts, in the same order as the file above
    generated_cifs_fname = "../out/cif_model_32.evalcifs-sg.pkl.gz"
    gen_attempts = 3
    out_fname = "../out/cif_model_32.evalresults-sg-match.csv"

    with gzip.open(true_cifs_fname, "rb") as f:
        true_cifs = pickle.load(f)

    with gzip.open(generated_cifs_fname, "rb") as f:
        generated_cifs = pickle.load(f)

    struct_matcher = StructureMatcher(
        ltol=0.2, stol=0.3, angle_tol=5, primitive_cell=True, scale=True,
        attempt_supercell=False, comparator=ElementComparator()
    )

    results = {
        "formula": [],
        "sg": [],
    }
    for k in range(gen_attempts):
        results[f"gen{k+1}_match"] = []

    errors = []

    for i, true_cif in tqdm(enumerate(true_cifs), total=len(true_cifs)):
        true_cif = clean_cif(true_cif)

        true_struct = Structure.from_str(true_cif, fmt="cif")

        results["formula"].append(extract_data_formula(true_cif))
        results["sg"].append(extract_space_group_symbol(true_cif))

        generated = generated_cifs[i]
        for k in range(gen_attempts):
            if type(generated) == list:
                gen_cif = generated[k]
            else:
                # if generated is not a list, then `gen_attempts` should be 1, and `generated` is a CIF
                gen_cif = generated

            gen_cif = clean_cif(gen_cif)

            try:
                # try to calculate the implied volume, to weed out very bad generations;
                #  an exception will be thrown if a value is missing, or the volume is nonsensical
                a = extract_numeric_property(gen_cif, "_cell_length_a")
                b = extract_numeric_property(gen_cif, "_cell_length_b")
                c = extract_numeric_property(gen_cif, "_cell_length_c")
                alpha = extract_numeric_property(gen_cif, "_cell_angle_alpha")
                beta = extract_numeric_property(gen_cif, "_cell_angle_beta")
                gamma = extract_numeric_property(gen_cif, "_cell_angle_gamma")
                get_unit_cell_volume(a, b, c, alpha, beta, gamma)

                gen_struct = Structure.from_str(gen_cif, fmt="cif")
                is_match = struct_matcher.fit(true_struct, gen_struct)
            except Exception as e:
                errors.append(str(e))
                is_match = False

            results[f"gen{k+1}_match"].append(1 if is_match else 0)

    df = pd.DataFrame(results)
    # add a column with the sum of gen columns, containing the total
    #  number of generations that match the true structure
    df["n_matching"] = df.loc[:, "gen1_match":f"gen{gen_attempts}_match"].sum(axis=1)
    # add a boolean column indicating whether the row has at least 1 generation
    #   that matches the true structure
    df["matches_at_least_once"] = df["n_matching"] >= 1
    # add a boolean column indicating whether the row has all k generations
    #  matching the true structure
    df["all_match"] = df["n_matching"] == gen_attempts
    df.to_csv(out_fname, index=False)

    for err in errors:
        print(f"error: {err}")

    print(f"fraction with at least 1 match within {gen_attempts} attempts: {df['matches_at_least_once'].mean():.3f}")
    print(f"fraction with all {gen_attempts} attempts matching: {df['all_match'].mean():.3f}")
    for k in range(gen_attempts):
        print(f"fraction matched on attempt {k+1}: {df[f'gen{k+1}_match'].mean():.3f}")
