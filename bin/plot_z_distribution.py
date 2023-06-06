import gzip
import pickle
from tqdm import tqdm
from lib import extract_formula_units
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")


if __name__ == '__main__':

    cifs_fname = "../out/orig_cifs_mp_2022_04_12+oqmd_v1_5+nomad_2023_04_30__comp-sg_augm.pkl.gz"

    with gzip.open(cifs_fname, "rb") as f:
        cifs = pickle.load(f)

    z_to_cif_count = {}
    cutoff = 8

    for cif in tqdm(cifs):
        z = extract_formula_units(cif)
        if z > cutoff:
            z = 9
        if z not in z_to_cif_count:
            z_to_cif_count[z] = 0
        z_to_cif_count[z] += 1

    for entry in sorted(z_to_cif_count.items(), key=lambda k: k[0]):
        if entry[0] == 9:
            print(f">8: {entry[1]:,}")
        else:
            print(f"{entry[0]}: {entry[1]:,}")

    X = list(range(1, 10))
    heights = []
    for x in X:
        heights.append(z_to_cif_count[x])

    plt.bar(X, heights)
    plt.title("Occurrence Counts of Z in the Dataset")
    plt.xticks(ticks=X, labels=["1", "2", "3", "4", "5", "6", "7", "8", ">8"])
    plt.xlabel("Z (formula units)")
    plt.ylabel("CIF count")
    plt.show()
