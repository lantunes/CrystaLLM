import gzip
import pickle
from pymatgen.core import Composition, Element
from tqdm import tqdm
from lib import extract_formula_nonreduced, get_cif_tokenizer
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")


if __name__ == '__main__':

    cifs_fname = "../out/orig_cifs_mp_2022_04_12+oqmd_v1_5+nomad_2023_04_30__comp-sg_augm.pkl.gz"
    out_fname = "../out/atom_distribution.pdf"

    atoms = get_cif_tokenizer(symmetrized=True).atoms()

    with gzip.open(cifs_fname, "rb") as f:
        cifs = pickle.load(f)

    atom_to_cif_count = {Element(atom): 0 for atom in atoms}

    for cif in tqdm(cifs):
        formula = extract_formula_nonreduced(cif)
        comp = Composition(formula)
        for elem in comp.elements:
            atom_to_cif_count[elem] += 1

    print("most represented atoms:")
    count_map = {e: count for e, count in atom_to_cif_count.items()}
    most_common = []
    for atom, count in sorted(count_map.items(), key=lambda x: x[1], reverse=True):
        print(f"atom: {atom}, count: {count}")
        most_common.append((atom, count))

    print(sorted({e.Z for e in atom_to_cif_count}))

    Zs = sorted([elem.Z for elem in atom_to_cif_count])
    heights = [atom_to_cif_count[Element.from_Z(Z)] for Z in Zs]

    plt.rcParams["figure.figsize"] = (15, 5)

    plt.bar(Zs, heights)
    plt.title("Occurrence Counts of Atoms in the Dataset")
    plt.xticks(Zs[::2])
    plt.xlabel("atomic number")
    plt.ylabel("CIF count")
    t = "\n".join([f"atom: {a}, count: {c:,}" for a, c in most_common][0:3])
    plt.text(0.8, 0.8, "most common:\n" + t, transform=plt.gca().transAxes)

    plt.savefig(out_fname)
    plt.show()
