import pandas as pd
from pymatgen.io.cif import CifWriter
from tqdm import tqdm
import gzip

try:
    import cPickle as pickle
except ImportError:
    import pickle

import warnings
warnings.filterwarnings("ignore")


if __name__ == '__main__':
    # NOTE: this file is located at s3://crystalgpt-data/matproj_all_2022_04_12.pkl
    fname = "../out/matproj_all_2022_04_12.pkl"
    out_file = "../out/orig_cifs_mp_2022_04_12.pkl.gz"
    symmetrize = True
    symprec = 0.1

    df = pd.read_pickle(fname)

    cifs = []

    for i in tqdm(range(len(df['structure']))):
        try:
            struct = df['structure'][i]
            mpid = df['material_id'][i]
            cif = str(CifWriter(struct, symprec=symprec if symmetrize else None))
            comment = f"# MP Entry {mpid} {struct.composition.reduced_formula} " \
                      f"{struct.composition.formula.replace(' ', '')}\n"
            cif = comment + cif
            cifs.append(cif)
        except Exception:
            pass

    print(f"CIFs generated: {len(cifs):,}")

    with gzip.open(out_file, 'wb') as f:
        pickle.dump(cifs, f, protocol=pickle.HIGHEST_PROTOCOL)
