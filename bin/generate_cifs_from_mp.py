import pandas as pd
from pymatgen.io.cif import CifWriter
from tqdm import tqdm

try:
    import cPickle as pickle
except ImportError:
    import pickle


if __name__ == '__main__':
    # NOTE: this file is located at s3://crystalgpt-data/matproj_all_2022_04_12.pkl
    fname = "../out/matproj_all_2022_04_12.pkl"
    out_file = "../out/matproj_all_2022_04_12.cif_nosymm.pkl"
    symmetrize = False

    df = pd.read_pickle(fname)

    cifs = []

    for i in tqdm(range(len(df['structure']))):
        struct = df['structure'][i]
        cif = CifWriter(struct, symprec=0.001 if symmetrize else None)
        cifs.append(str(cif))

    with open(out_file, 'wb') as f:
        pickle.dump(cifs, f, protocol=pickle.HIGHEST_PROTOCOL)
