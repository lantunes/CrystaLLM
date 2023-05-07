import gzip
from tqdm import tqdm

try:
    import cPickle as pickle
except ImportError:
    import pickle

import warnings
warnings.filterwarnings("ignore")


def load_cifs(fname):
    with gzip.open(fname, "rb") as f:
       cifs = pickle.load(f)
    return cifs


def add_cifs(cifs, prefix, all_cifs, all_ids):
    for cif in tqdm(cifs):
        # extract id from the first comment in the CIF
        id = prefix + "_" + cif.split("\n")[0].split(" ")[3]
        if id in all_ids:
            raise Exception(f"ID exists: {id}")
        all_ids.add(id)
        all_cifs.append((id, cif))


if __name__ == '__main__':
    mp_cifs_fname = "../out/orig_cifs_mp_2022_04_12.pkl.gz"
    oqmd_cifs_fname = "../out/orig_cifs_oqmd_v1_5.pkl.gz"
    nomad_cifs_fname = "../out/orig_cifs_nomad_2023_04_30.pkl.gz"
    out_fname = "../out/orig_cifs_mp_2022_04_12+oqmd_v1_5+nomad_2023_04_30.pkl.gz"

    print("loading CIFs...")
    mp_cifs = load_cifs(mp_cifs_fname)
    oqmd_cifs = load_cifs(oqmd_cifs_fname)
    nomad_cifs = load_cifs(nomad_cifs_fname)

    ids = set()
    ids_and_cifs = []

    print("adding CIFs...")
    add_cifs(mp_cifs, "MP", ids_and_cifs, ids)
    add_cifs(oqmd_cifs, "OQMD", ids_and_cifs, ids)
    add_cifs(nomad_cifs, "NOMAD", ids_and_cifs, ids)

    assert len(ids) == len(ids_and_cifs), "every ID should be unique"

    assert len(ids_and_cifs) == (len(mp_cifs) + len(oqmd_cifs) + len(nomad_cifs)), "all CIFs should be included"

    print(f"CIFs total count: {len(ids_and_cifs)}")

    print("writing CIFs...")

    with gzip.open(out_fname, "wb") as f:
        pickle.dump(ids_and_cifs, f, protocol=pickle.HIGHEST_PROTOCOL)
