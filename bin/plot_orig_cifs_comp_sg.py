import gzip
import pickle
import numpy as np
from tqdm import tqdm
from pymatgen.core import Composition
from lib import extract_formula_nonreduced, extract_numeric_property
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import warnings
warnings.filterwarnings("ignore")


if __name__ == '__main__':

    cifs_fname = "../out/orig_cifs_mp_2022_04_12+oqmd_v1_5+nomad_2023_04_30__comp-sg.pkl.gz"

    with gzip.open(cifs_fname, "rb") as f:
        cifs = pickle.load(f)

    elements_space_group_nums = {}

    for id, cif in tqdm(cifs):
        formula = extract_formula_nonreduced(cif)
        comp = Composition(formula)
        sg_num = extract_numeric_property(cif, "_symmetry_Int_Tables_number", numeric_type=int)

        for elem in comp.elements:
            if elem.Z > 0:
                key = (sg_num, elem.Z)
                if key not in elements_space_group_nums:
                    elements_space_group_nums[key] = 0
                elements_space_group_nums[key] += 1

    max_sg_num = float("-inf")
    max_Z = float("-inf")

    for sg_num, Z in elements_space_group_nums.keys():
        if sg_num > max_sg_num:
            max_sg_num = sg_num
        if Z > max_Z:
            max_Z = Z

    X = [[0. for _ in range(0, max_sg_num)] for _ in range(0, max_Z)]

    for sg_num, Z in elements_space_group_nums.keys():
        X[Z-1][sg_num-1] = np.log1p(elements_space_group_nums[(sg_num, Z)])

    plt.figure()
    ax = plt.gca()
    im = ax.imshow(X, origin="lower", cmap="Blues")
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    X = np.array(X)
    x_positions = np.arange(0, X.shape[1], 10)
    x_labels = np.arange(1, X.shape[1] + 1, 10)
    y_positions = np.arange(0, X.shape[0], 10)
    y_labels = np.arange(1, X.shape[0] + 1, 10)
    ax.set_xticks(x_positions, x_labels)
    ax.set_yticks(y_positions, y_labels)
    ax.set_xlabel("Space group number")
    ax.set_ylabel("Atomic number")

    plt.colorbar(im, cax=cax)
    plt.show()
