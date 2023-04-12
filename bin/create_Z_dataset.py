import gzip
import re
from tqdm import tqdm
try:
    import cPickle as pickle
except ImportError:
    import pickle

import warnings
warnings.filterwarnings("ignore")


if __name__ == '__main__':
    fname = "../out/oqmd_v1_5_matproj_all_2022_04_12.cif_semisymm.pkl.gz"
    out_file = "../out/oqmd_v1_5_matproj_all_2022_04_12.cif_semisymm_Z.pkl.gz"

    with gzip.open(fname, "rb") as f:
        cifs_raw = pickle.load(f)

    modified_cifs = []
    pattern = r"_chemical_formula_sum\s+(.+)\n"
    pattern_2 = r"(data_)(.*?)(\n)"

    for cif in tqdm(cifs_raw):
        match = re.search(pattern, cif)
        if match:
            chemical_formula = match.group(1)
            chemical_formula = chemical_formula.replace("'", "").replace(" ", "")

            modified_cif = re.sub(pattern_2, r'\1' + chemical_formula + r'\3', cif)

            modified_cifs.append(modified_cif)
        else:
            raise Exception(f"Chemical formula not found {cif}")

    with gzip.open(out_file, "wb") as f:
        pickle.dump(modified_cifs, f)
