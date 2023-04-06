import gzip
from tqdm import tqdm
import re
try:
    import cPickle as pickle
except ImportError:
    import pickle


if __name__ == '__main__':
    fname = "../out/oqmd_v1_5_matproj_all_2022_04_12.cif.pkl.gz"
    out_file = "../out/oqmd_v1_5_matproj_all_2022_04_12.cif_semisymm.pkl.gz"

    with gzip.open(fname, "rb") as f:
        cifs_raw = pickle.load(f)

    modified_cifs = []

    for cif_raw in tqdm(cifs_raw):
        pattern = re.compile(r"(_symmetry_equiv_pos_as_xyz\n)(.*?)(?=\n(?:\S| \S))", re.DOTALL)
        replacement = r"\1  1  'x, y, z'"
        modified_cif = re.sub(pattern, replacement, cif_raw)

        modified_cifs.append(modified_cif)

    with gzip.open(out_file, "wb") as f:
        pickle.dump(modified_cifs, f)
