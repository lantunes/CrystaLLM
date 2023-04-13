import gzip
import re
from lib import get_atomic_props_block
from pymatgen.core.structure import Structure
from tqdm import tqdm
try:
    import cPickle as pickle
except ImportError:
    import pickle

import warnings
warnings.filterwarnings("ignore")


if __name__ == '__main__':
    fname = "../out/oqmd_v1_5_matproj_all_2022_04_12.cif_semisymm_Z.pkl.gz"
    out_file = "../out/oqmd_v1_5_matproj_all_2022_04_12.cif_semisymm_Z_props.pkl.gz"
    oxi = False  # whether the CIFs to modify contain oxidation state information

    with gzip.open(fname, "rb") as f:
        cifs_raw = pickle.load(f)

    modified_cifs = []

    for cif in tqdm(cifs_raw):
        struct = Structure.from_str(cif, fmt="cif")

        block = get_atomic_props_block(composition=struct.composition, oxi=oxi)

        # the hypothesis is that the atomic properties should be the first thing
        #  that the model must learn to associate with the composition, since
        #  they will determine so much of what follows in the file
        pattern = r"_symmetry_space_group_name_H-M"
        match = re.search(pattern, cif)

        if match:
            start_pos = match.start()
            modified_cif = cif[:start_pos] + block + "\n" + cif[start_pos:]
            modified_cifs.append(modified_cif)
        else:
            raise Exception(f"Pattern not found: {cif}")

    with gzip.open(out_file, "wb") as f:
        pickle.dump(modified_cifs, f, protocol=pickle.HIGHEST_PROTOCOL)
