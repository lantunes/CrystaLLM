import gzip
import pickle
from tqdm import tqdm
from lib import extract_numeric_property
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")


if __name__ == '__main__':

    cifs_fname = "../out/orig_cifs_mp_2022_04_12+oqmd_v1_5+nomad_2023_04_30__comp-sg_augm.pkl.gz"
    out_fname = "../out/spacegroup_dist.pdf"

    with gzip.open(cifs_fname, "rb") as f:
        cifs = pickle.load(f)

    sg_to_cif_count = [0 for _ in range(0, 230)]

    for cif in tqdm(cifs):
        sg = extract_numeric_property(cif, "_symmetry_Int_Tables_number", int)
        sg_to_cif_count[sg-1] += 1

    print("most represented space groups:")
    count_map = {i+1: count for i, count in enumerate(sg_to_cif_count)}
    most_common = []
    for i, count in sorted(count_map.items(), key=lambda x: x[1], reverse=True):
        print(f"sg: {i}, count: {count}")
        most_common.append((i, count))

    plt.rcParams["figure.figsize"] = (15, 5)

    plt.bar(list(range(1, 231)), sg_to_cif_count)
    plt.title("Occurrence Counts of Space Groups in the Dataset")
    plt.xticks(list(range(1, 231, 10)))
    plt.xlabel("space group number")
    plt.ylabel("CIF count")
    t = "\n".join([f"space group: {sg}, count: {c:,}" for sg, c in most_common][0:3])
    plt.text(0.1, 0.8, "most common:\n" + t, transform=plt.gca().transAxes)

    plt.savefig(out_fname)
    plt.show()
