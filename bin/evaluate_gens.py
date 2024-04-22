import sys, os
import argparse
import tarfile
import queue
import multiprocessing as mp

from tqdm import tqdm
import numpy as np
import pandas as pd
import re

from pymatgen.core.structure import Composition
from pymatgen.io.cif import CifParser

script_dir = os.path.dirname(os.path.abspath(__file__))
crystallm_dir = os.path.abspath(os.path.join(script_dir, '..'))  
sys.path.append(crystallm_dir)

from crystallm import (
    CIFTokenizer,
    bond_length_reasonableness_score,
    extract_data_formula,
    extract_formula_based_on_coords,
    extract_numeric_property,
    extract_space_group_symbol,
    extract_volume,
    get_unit_cell_volume,
    is_atom_site_multiplicity_consistent,
    is_space_group_consistent,
    is_formula_consistent_based_on_coords,
    is_sensible,
    is_valid,
    replace_symmetry_operators,
    load_labels,
    # evaluate_structure_similarity,
    check_indices_validity,
    segregate_structure,
    is_molecule,
    is_fully_connected,
    is_adsorbed,
    validate_ads_slab,
    match_adsorbate_composition,
    match_slab_composition_ratio,
    load_and_evaluate_similarity
)

import warnings
warnings.filterwarnings("ignore")

def read_generated_cifs_with_sids(input_path):
    generated_cifs = []
    generated_sids = []
    with tarfile.open(input_path, "r:gz") as tar:
        for member in tqdm(tar.getmembers(), desc="extracting generated CIFs..."):
            f = tar.extractfile(member)
            if f is not None:
                cif = f.read().decode("utf-8")
                generated_cifs.append(cif)
                generated_sids.append(member.name.split('/')[-1].split('.')[0])
    return generated_cifs, generated_sids



# def eval_structure(generated_cifs, generated_sids, length_lo, length_hi, angle_lo, angle_hi, traj_dir):
#     tokenizer = CIFTokenizer()
    
#     n_sensible = 0
#     n_atom_site_multiplicity_consistent = 0
#     n_space_group_consistent = 0   
#     n_formula_consistent = 0
#     #bond_length_reasonableness_scores = []
#     result = {}
#     for cif, sid in zip(generated_cifs, generated_sids):
#         sensible = is_sensible(cif, length_lo, length_hi, angle_lo, angle_hi)
#         if sensible:
#             n_sensible += 1

#         gen_len = len(tokenizer.tokenize_cif(cif))
        
#         space_group_symbol = extract_space_group_symbol(cif)
#         if space_group_symbol is not None and space_group_symbol != "P 1":
#             cif = replace_symmetry_operators(cif, space_group_symbol)
    
#         multi_consistency = is_atom_site_multiplicity_consistent(cif)
#         if multi_consistency:
#             n_atom_site_multiplicity_consistent += 1

#         sg_consistency = is_space_group_consistent(cif)
#         if sg_consistency:
#             n_space_group_consistent += 1        

#         formula_consistency = is_formula_consistent_based_on_coords(cif)
#         if formula_consistency:
#             n_formula_consistent += 1

#         score = bond_length_reasonableness_score(cif, tolerance=0.3, h_factor=2.5)
#         #bond_length_reasonableness_scores.append(score)

#         a = extract_numeric_property(cif, "_cell_length_a")
#         b = extract_numeric_property(cif, "_cell_length_b")
#         c = extract_numeric_property(cif, "_cell_length_c")
#         alpha = extract_numeric_property(cif, "_cell_angle_alpha")
#         beta = extract_numeric_property(cif, "_cell_angle_beta")
#         gamma = extract_numeric_property(cif, "_cell_angle_gamma")
#         implied_vol = get_unit_cell_volume(a, b, c, alpha, beta, gamma)

#         gen_vol = extract_volume(cif)
#         data_formula = extract_data_formula(cif)
#         data_formula = re.sub(r"_miller_\d+", "", data_formula)

#         data_formula_coords = extract_formula_based_on_coords(cif)
#         breakpoint()
#         ## load label structure and energy
#         label_structure, label_energy = load_labels(sid, traj_dir)
        
#         ## structure comparison
#         structure_rmsd = evaluate_structure_similarity(cif, label_structure)

#         ## energy comparison
#         # predicted_energy = evaluate_energy(cif)
#         # energy_error = abs(predicted_energy - label_energy)

#         result[sid] = {
#             "sensible": sensible,
#             "atom_site_multiplicity_consistent": multi_consistency,
#             "space_group_consistent": sg_consistency,
#             "formula_consistent": formula_consistency,
#             "bond_length_reasonableness_score": score,
#             "implied_volume": implied_vol,
#             "generated_volume": gen_vol,
#             "data_formula_file": data_formula,
#             "data_formula_coords": data_formula_coords,
#             "generated_length": gen_len,
#             "structure_rmsd": structure_rmsd,
#             # "predicted_energy": predicted_energy,
#             # "label_energy": label_energy,
#             # "energy_error": energy_error
#         }
#         breakpoint()

        
#     return result


def eval_structure(generated_cifs, generated_sids, metadata_path, traj_dir, save_path):

    tokenizer = CIFTokenizer()
    metadata = pd.read_pickle(metadata_path)
    
    # n_valid_adsorbate = 0
    # n_connected_slab = 0
    # n_adsorbed = 0

    result = {}
    for cif, sid in zip(generated_cifs, generated_sids):
        try:
            parser = CifParser.from_str(cif)
            adslab_structure = parser.get_structures()[0]
        except:
            print(f"Error reading CIF file: {sid}")
            continue
        # segregate the adslab into adsorbate and slab
        unq_sid = re.match(r'random\d+', sid).group()
        bulk_symbols = metadata[unq_sid]["bulk_symbols"]
        ads_symbols = metadata[unq_sid]["ads_symbols"]
        ads_symbols = ads_symbols.replace("*", "") #post-processing
        # breakpoint()
        bulk_indices, ads_indices = segregate_structure(adslab_structure, bulk_symbols, ads_symbols)
        # breakpoint()
        '''
        CIF file information
        '''
        gen_len = len(tokenizer.tokenize_cif(cif))
        space_group_symbol = extract_space_group_symbol(cif)
        if space_group_symbol is not None and space_group_symbol != "P 1":
            cif = replace_symmetry_operators(cif, space_group_symbol)
        
        a = extract_numeric_property(cif, "_cell_length_a")
        b = extract_numeric_property(cif, "_cell_length_b")
        c = extract_numeric_property(cif, "_cell_length_c")
        alpha = extract_numeric_property(cif, "_cell_angle_alpha")
        beta = extract_numeric_property(cif, "_cell_angle_beta")
        gamma = extract_numeric_property(cif, "_cell_angle_gamma")
        implied_vol = get_unit_cell_volume(a, b, c, alpha, beta, gamma)

        gen_vol = extract_volume(cif)
        data_formula = extract_data_formula(cif)
        if "_miller_" in data_formula:
            data_formula = re.sub(r"_miller_\d+", "", data_formula)

        data_formula_coords = extract_formula_based_on_coords(cif)

        '''
        validity check
        1. check if the adsorbate is valid
        2. check if the slab is fully connected
        3. check if the adsorbate is adsorbed
        4. check the bond length reasonableness
        '''
        ads_valid = is_molecule(adslab_structure, ads_indices)
        slab_connected = is_fully_connected(adslab_structure, bulk_indices)
        adsorbed = is_adsorbed(adslab_structure, ads_indices)
        score = bond_length_reasonableness_score(cif, tolerance=0.3, h_factor=2.5)
        
        '''
        accuracy check
        1. check if the adsorbate composition is correct
        2. check if the slab composition ratio is correct
        3. calculate the structural similarity (RMSD)
        '''
        ads_compos_match, ads_cif_comps, ads_meta_comps = match_adsorbate_composition(adslab_structure, ads_indices, ads_symbols)
        # breakpoint()
        slab_compos_match, slab_cif_comps, slab_meta_comps = match_slab_composition_ratio(adslab_structure, bulk_indices, bulk_symbols)
        rmsd = load_and_evaluate_similarity(sid, traj_dir, adslab_structure)

        result[sid] = {
            "generated_length": gen_len,
            "implied_volume": implied_vol,
            "generated_volume": gen_vol,
            "data_formula_file": data_formula,
            "data_formula_coords": data_formula_coords,

            "adsorbate_valid": ads_valid,
            "slab_connected": slab_connected,
            "adsorbed": adsorbed,
            "bond_length_score": score,

            "adsorbate_compos_match": ads_compos_match,
            "adsorbate_compos": (ads_cif_comps, ads_meta_comps),
            "slab_compos_match": slab_compos_match,
            "slab_compos": (slab_cif_comps, slab_meta_comps),
            "structure_rmsd": rmsd
        }
        # breakpoint()

    if save_path:
        result_df = pd.DataFrame(result).T
        result_df.to_csv(save_path)

    return result

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate generated structures.")
    parser.add_argument("gen_cifs",
                        help="Path to the .tar.gz file containing the generated CIF files.")
    parser.add_argument("--length_lo", required=False, default=0.5, type=float,
                        help="The smallest cell length allowable for the sensibility check")
    parser.add_argument("--length_hi", required=False, default=1000., type=float,
                        help="The largest cell length allowable for the sensibility check")
    parser.add_argument("--angle_lo", required=False, default=10., type=float,
                        help="The smallest cell angle allowable for the sensibility check")
    parser.add_argument("--angle_hi", required=False, default=170., type=float,
                        help="The largest cell angle allowable for the sensibility check")
    parser.add_argument("--traj_dir", required=False, default="traj",
                        help="Path to the directory containing trajectories")
    parser.add_argument("--metadata_path", required=False, default="metadata.pkl",
                        help="Path to the metadata file")
    parser.add_argument("--save_path", required=False, default="eval_results.csv",
                        help="Path to save the evaluation results")
    args = parser.parse_args()

    gen_cifs_path = args.gen_cifs
    length_lo = args.length_lo
    length_hi = args.length_hi
    angle_lo = args.angle_lo
    angle_hi = args.angle_hi
    traj_dir = args.traj_dir
    metadata_path = args.metadata_path
    save_path = args.save_path

    cifs, sids = read_generated_cifs_with_sids(gen_cifs_path)
    # breakpoint()
    #structure_result = eval_structure(cifs, sids, length_lo, length_hi, angle_lo, angle_hi, traj_dir)
    structure_result = eval_structure(cifs, sids, metadata_path, traj_dir, save_path)