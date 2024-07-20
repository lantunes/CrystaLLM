import os
import argparse
import gzip
import pickle
import tarfile
import re
from pymatgen.core import Composition, Structure
from pymatgen.analysis.structure_matcher import StructureMatcher
from crystallm import extract_data_formula, is_valid
from postprocess import postprocess
from tqdm import tqdm

import warnings
warnings.filterwarnings("ignore")


def read_gen_cifs(gen_cifs_path):
    cifs = []
    if gen_cifs_path.endswith(".tar.gz"):
        with tarfile.open(gen_cifs_path, "r:gz") as tar:
            for member in tar.getmembers():
                if member.isfile() and member.name.endswith(".cif"):
                    cif = tar.extractfile(member).read().decode()
                    cifs.append(cif)
    else:
        for filename in os.listdir(gen_cifs_path):
            if filename.endswith(".cif"):
                file_path = os.path.join(gen_cifs_path, filename)
                with open(file_path, "r") as file:
                    cif = file.read()
                    cifs.append(cif)
    return cifs


"""
Given a path to a directory containing generated post-processed CIFs, and
a path to a set of CIFs used for training a model, this script does the following:
1. prints the number of given CIFs which are valid
2. prints the number of given CIFs which are unique
3. prints the number of given CIFs which are novel relative to the training CIFs
4. copies the novel CIFs to a specified path
"""
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Count the number of valid, unique, and novel CIFs.")
    parser.add_argument("gen_cifs",
                        help="Path to the directory or .tar.gz file containing the "
                             "generated post-processed CIF files.")
    parser.add_argument("--base",
                        help="Path to the file containing the set of pre-processed CIFs used for training the model, "
                             "as a .pkl.gz file containing a Python list of 2-tuples: (ID, CIF).")
    parser.add_argument("--out",
                        help="Path to the directory where the novel CIFs should be copied.")
    # the default StructureMatcher settings are based on the MatterGen paper https://arxiv.org/abs/2312.03687
    parser.add_argument("--ltol", required=False, default=0.2, type=float,
                        help="The 'ltol' argument for the pymatgen StructureMatcher")
    parser.add_argument("--stol", required=False, default=0.3, type=float,
                        help="The 'stol' argument for the pymatgen StructureMatcher")
    parser.add_argument("--angle_tol", required=False, default=5., type=float,
                        help="The 'angle_tol' argument for the pymatgen StructureMatcher")
    args = parser.parse_args()

    gen_cifs_path = args.gen_cifs
    base_cifs_path = args.base
    out_path = args.out
    ltol = args.ltol
    stol = args.stol
    angle_tol = args.angle_tol

    with gzip.open(base_cifs_path, "rb") as f:
        base_cifs = pickle.load(f)

    print(f"# base CIFs: {len(base_cifs):,}")

    base_comps = {}
    for _, cif in tqdm(base_cifs, desc="processing base CIFs..."):
        cell_composition = extract_data_formula(cif)
        formula = Composition(cell_composition).formula
        if formula not in base_comps:
            base_comps[formula] = []
        # strip out any leading or trailing spaces
        cif = re.sub(r"^[ \t]+|[ \t]+$", "", cif, flags=re.MULTILINE)
        cif = cif.replace("  ", " ")
        base_comps[formula].append(cif)

    print(f"reading generated CIFs from {gen_cifs_path}...")
    gen_cifs = read_gen_cifs(gen_cifs_path)

    struct_matcher = StructureMatcher(ltol=ltol, stol=stol, angle_tol=angle_tol)

    # keep only the valid CIFs
    valid_gen_cifs = []
    for gen_cif in gen_cifs:
        try:
            if is_valid(gen_cif, bond_length_acceptability_cutoff=1.0):
                valid_gen_cifs.append(gen_cif)
        except Exception:
            pass

    print(f"# valid CIFs: {len(valid_gen_cifs):,}")

    # get unique generated CIFs;
    #  create a map cell_comp->list of structs, where each struct is unique;
    #  the number of unique structs is the number of unique materials
    gen_comps = {}
    for gen_cif in tqdm(valid_gen_cifs, desc="processing generated CIFs..."):
        struct = Structure.from_str(gen_cif, fmt="cif")
        cell_composition = struct.composition.formula
        if cell_composition not in gen_comps:
            gen_comps[cell_composition] = [(struct, gen_cif)]
        else:
            # check that none of the structs match
            if not any([
                struct_matcher.fit(struct, existing)
                for existing, _ in gen_comps[cell_composition]
            ]):
                gen_comps[cell_composition].append((struct, gen_cif))

    # unnest structs
    gen_structs = [i for g in gen_comps.values() for i in g]

    print(f"# unique materials: {len(gen_structs):,}")

    # only unique structures can be novel; if we had 1 unique structure out of 100 generated, we wouldn't have
    #  100 novel structures--we'd have 1 (if it were indeed novel)

    novel_cifs = []
    novel_by_composition = 0

    for gen_struct, gen_cif in tqdm(gen_structs, desc="checking novelty..."):
        gen_formula = gen_struct.composition.formula

        if gen_formula not in base_comps:
            # this is a novel material as there is no composition match
            # print(f"novel by formula: {gen_struct.composition.reduced_formula}")
            novel_cifs.append(gen_cif)
            novel_by_composition += 1
        else:
            # for each existing CIF, post-process it and convert to structure and check structure match
            is_matched = False
            for existing_cif in base_comps[gen_formula]:
                existing_cif_post = postprocess(existing_cif, "N/A")
                existing_struct = Structure.from_str(existing_cif_post, fmt="cif")
                is_matched = struct_matcher.fit(gen_struct, existing_struct)
                if is_matched:
                    # print(f"not novel: {gen_struct.composition.reduced_formula}")
                    break
            if not is_matched:
                # print(f"novel: {gen_struct.composition.reduced_formula}")
                novel_cifs.append(gen_cif)

    print(f"# novel materials: {len(novel_cifs):,}")
    print(f"# novel materials by composition alone: {novel_by_composition:,}")

    if len(novel_cifs) > 0:
        if not os.path.exists(out_path):
            print(f"creating {out_path}...")
            os.makedirs(out_path)

        print(f"writing novel CIFs...")
        for i, novel_cif in enumerate(novel_cifs):
            output_file_path = os.path.join(out_path, f"novel_{i+1}.cif")
            with open(output_file_path, "w") as file:
                file.write(novel_cif)
