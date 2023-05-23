import gzip
from tqdm import tqdm
from lib import extract_formula_nonreduced, extract_space_group_symbol, extract_volume, extract_formula_units

try:
    import cPickle as pickle
except ImportError:
    import pickle

import warnings
warnings.filterwarnings("ignore")


if __name__ == '__main__':
    cifs_fname = "../out/orig_cifs_mp_2022_04_12+oqmd_v1_5+nomad_2023_04_30.pkl.gz"
    out_fname = "../out/orig_cifs_mp_2022_04_12+oqmd_v1_5+nomad_2023_04_30__comp-sg.pkl.gz"

    with gzip.open(cifs_fname, "rb") as f:
        cifs = pickle.load(f)

    lowest_vpfu = {}

    for id, cif in tqdm(cifs):
        formula = extract_formula_nonreduced(cif)
        space_group = extract_space_group_symbol(cif)
        formula_units = extract_formula_units(cif)
        if formula_units == 0:
            formula_units = 1
        vpfu = extract_volume(cif) / formula_units

        key = (formula, space_group)
        if key not in lowest_vpfu:
            lowest_vpfu[key] = (id, cif, vpfu)
        else:
            existing_vpfu = lowest_vpfu[key][2]
            if vpfu < existing_vpfu:
                lowest_vpfu[key] = (id, cif, vpfu)

    selected_entries = [(id, cif) for id, cif, _ in lowest_vpfu.values()]

    print(f"number of entries to write: {len(selected_entries)}")

    with gzip.open(out_fname, "wb") as f:
        pickle.dump(selected_entries, f, protocol=pickle.HIGHEST_PROTOCOL)
