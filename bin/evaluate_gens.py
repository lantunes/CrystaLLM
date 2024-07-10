import sys, os
import argparse
import tarfile
import queue
import multiprocessing as mp

from tqdm import tqdm
import numpy as np
import pandas as pd
import re

from pymatgen.core.structure import Composition, Structure
from pymatgen.io.cif import CifParser
from pymatgen.analysis import structure_analyzer

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
    extract_ads_bulk_symbols,  
    get_unit_cell_volume,
    replace_symmetry_operators,
    segregate_structure,
    is_molecule,
    is_bond_length_reasonable,
    is_adsorbed,
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



def eval_structure(generated_cifs, generated_sids, metadata_path, traj_dir, save_path):
    # generated_cifs --> list of CIF strings
    tokenizer = CIFTokenizer()
    metadata = pd.read_pickle(metadata_path)
    
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
        bulk_structure = Structure.from_sites([adslab_structure[i] for i in bulk_indices])
        symmetry_finder = structure_analyzer.SpacegroupAnalyzer(bulk_structure)
        space_group = symmetry_finder.get_space_group_symbol()

        '''
        CIF file information
        '''
        gen_len = len(tokenizer.tokenize_cif(cif))
        try:
            space_group_symbol = extract_space_group_symbol(cif)
            if space_group_symbol is not None and space_group_symbol != "P 1":
                cif = replace_symmetry_operators(cif, space_group_symbol)
        except:
            pass 

        a = extract_numeric_property(cif, "_cell_length_a")
        b = extract_numeric_property(cif, "_cell_length_b")
        c = extract_numeric_property(cif, "_cell_length_c")
        alpha = extract_numeric_property(cif, "_cell_angle_alpha")
        beta = extract_numeric_property(cif, "_cell_angle_beta")
        gamma = extract_numeric_property(cif, "_cell_angle_gamma")
        implied_vol = get_unit_cell_volume(a, b, c, alpha, beta, gamma)
        try:
            gen_vol = extract_volume(cif)
        except:
            gen_vol = None
        data_formula = extract_data_formula(cif)
        if "_miller_" in data_formula:
            data_formula = re.sub(r"_miller_\d+", "", data_formula)

        data_formula_coords = extract_formula_based_on_coords(cif)

        '''
        validity check
        1. check if the adsorbate is a single, connected molecule
        2. check if the slab has reasonable connectivity
        3. check if the adsorbate is adsorbed
        4. check if the adsorbate composition matches the given input
        5. check if the slab composition ratio matches the given input
        '''
        ads_valid = is_molecule(adslab_structure, ads_indices)
        slab_valid = (is_bond_length_reasonable(adslab_structure, bulk_indices) == 1)
        adsorbed = is_adsorbed(adslab_structure, ads_indices)
        score = bond_length_reasonableness_score(cif, tolerance=0.3, h_factor=2.5)
        ads_compos_match, ads_cif_comps, ads_meta_comps = match_adsorbate_composition(adslab_structure, ads_indices, ads_symbols)
        slab_compos_match, slab_cif_comps, slab_meta_comps = match_slab_composition_ratio(adslab_structure, bulk_indices, bulk_symbols)
        
        # additional structure comparison
        rmsd = load_and_evaluate_similarity(sid, traj_dir, adslab_structure)

        result[sid] = {
            # general information
            "generated_length": gen_len,
            "implied_volume": implied_vol,
            "generated_volume": gen_vol,
            "data_formula_file": data_formula,
            "data_formula_coords": data_formula_coords,
            # validity check
            "adsorbate_valid": ads_valid,
            "slab_valid": slab_valid,
            "adsorbed": adsorbed,
            "bond_length_score": score,
            "adsorbate_compos_match": ads_compos_match,
            "slab_compos_match": slab_compos_match,
            # additional info for record
            "adsorbate_compos": (ads_cif_comps, ads_meta_comps),
            "slab_compos": (slab_cif_comps, slab_meta_comps),
            "structure_rmsd": rmsd
        }
        # breakpoint()

    if save_path:
        result_df = pd.DataFrame(result).T
        result_df.to_csv(save_path)

    return result


def self_eval_structure(generated_cifs, save_path):

    tokenizer = CIFTokenizer()
    # metadata = pd.read_pickle(metadata_path)
    
    result = {}
    for cif in generated_cifs:
        try:
            parser = CifParser.from_str(cif)
            adslab_structure = parser.get_structures()[0]
        except:
            print(f"Error reading CIF file")
            continue
        # segregate the adslab into adsorbate and slab
        # NEED TO CHANGE THIS PART!!!
        # find the part starting with 'data_'
        
        bulk_symbols, ads_symbols = extract_ads_bulk_symbols(cif)
        # breakpoint()
        bulk_indices, ads_indices = segregate_structure(adslab_structure, bulk_symbols, ads_symbols)
        # breakpoint()
        bulk_structure = Structure.from_sites([adslab_structure[i] for i in bulk_indices])
        symmetry_finder = structure_analyzer.SpacegroupAnalyzer(bulk_structure)
        space_group = symmetry_finder.get_space_group_symbol()

        '''
        CIF file information
        '''
        gen_len = len(tokenizer.tokenize_cif(cif))
        try:
            space_group_symbol = extract_space_group_symbol(cif)
            if space_group_symbol is not None and space_group_symbol != "P 1":
                cif = replace_symmetry_operators(cif, space_group_symbol)
        except:
            pass 

        a = extract_numeric_property(cif, "_cell_length_a")
        b = extract_numeric_property(cif, "_cell_length_b")
        c = extract_numeric_property(cif, "_cell_length_c")
        alpha = extract_numeric_property(cif, "_cell_angle_alpha")
        beta = extract_numeric_property(cif, "_cell_angle_beta")
        gamma = extract_numeric_property(cif, "_cell_angle_gamma")
        implied_vol = get_unit_cell_volume(a, b, c, alpha, beta, gamma)
        try:
            gen_vol = extract_volume(cif)
        except:
            gen_vol = None
        # data_formula = extract_data_formula(cif)
        # if "_miller_" in data_formula:
        #     data_formula = re.sub(r"_miller_\d+", "", data_formula)

        data_formula_coords = extract_formula_based_on_coords(cif)

        '''
        validity check
        1. check if the adsorbate is a single, connected molecule
        2. check if the slab has reasonable connectivity
        3. check if the adsorbate is adsorbed
        4. check if the adsorbate composition matches the given input
        5. check if the slab composition ratio matches the given input
        '''
        ads_valid = is_molecule(adslab_structure, ads_indices)
        slab_valid = (is_bond_length_reasonable(adslab_structure, bulk_indices) == 1)
        adsorbed = is_adsorbed(adslab_structure, ads_indices)
        score = bond_length_reasonableness_score(cif, tolerance=0.3, h_factor=2.5)
        ads_compos_match, ads_cif_comps, ads_meta_comps = match_adsorbate_composition(adslab_structure, ads_indices, ads_symbols)
        slab_compos_match, slab_cif_comps, slab_meta_comps = match_slab_composition_ratio(adslab_structure, bulk_indices, bulk_symbols)
        
        # additional structure comparison
        #rmsd = load_and_evaluate_similarity(sid, traj_dir, adslab_structure)
        key = bulk_symbols+"_"+ads_symbols
        result[key] = {
            # general information
            "generated_length": gen_len,
            "implied_volume": implied_vol,
            "generated_volume": gen_vol,
            # "data_formula_file": data_formula,
            "data_formula_coords": data_formula_coords,
            # validity check
            "adsorbate_valid": ads_valid,
            "slab_valid": slab_valid,
            "adsorbed": adsorbed,
            "bond_length_score": score,
            "adsorbate_compos_match": ads_compos_match,
            "slab_compos_match": slab_compos_match,
            # additional info for record
            "adsorbate_compos": (ads_cif_comps, ads_meta_comps),
            "slab_compos": (slab_cif_comps, slab_meta_comps),
            # "structure_rmsd": rmsd
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
    # parser.add_argument("--traj_dir", required=False, default="traj",
    #                     help="Path to the directory containing trajectories")
    # parser.add_argument("--metadata_path", required=False, default="metadata.pkl",
    #                     help="Path to the metadata file")
    parser.add_argument("--save_path", required=False, default="eval_results.csv",
                        help="Path to save the evaluation results")
    args = parser.parse_args()

    gen_cifs_path = args.gen_cifs
    # traj_dir = args.traj_dir
    # metadata_path = args.metadata_path
    save_path = args.save_path

    cifs, sids = read_generated_cifs_with_sids(gen_cifs_path)
    # breakpoint()
    # structure_result = eval_structure(cifs, sids, metadata_path, traj_dir, save_path)
    structure_result = self_eval_structure(cifs, save_path)