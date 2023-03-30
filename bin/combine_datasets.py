from tqdm import tqdm
try:
    import cPickle as pickle
except ImportError:
    import pickle


if __name__ == '__main__':
    files_to_combine = [
        "../out/oqmd_v1_5.cif_nosymm.pkl",
        "../out/matproj_all_2022_04_12.cif_nosymm.pkl"
    ]
    out_file = "../out/oqmd_v1_5_matproj_all_2022_04_12.cif_nosymm.pkl"

    all_cifs = []

    for fname in tqdm(files_to_combine):
        with open(fname, "rb") as f:
            cifs_raw = pickle.load(f)
        all_cifs.extend(cifs_raw)

    with open(out_file, 'wb') as f:
        pickle.dump(all_cifs, f, protocol=pickle.HIGHEST_PROTOCOL)
