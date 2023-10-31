from sklearn.manifold import TSNE

from lib import atom_vectors_from_csv
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np


def color_by_label(label):
    if label in ["P1", "P-1"]:
        # triclinic
        return "pink", "black"
    if label in ["P2", "P2_1", "C2", "Pm", "Pc", "Cm", "Cc", "P2/m",
                 "P2_1/m", "C2/m", "P2/c", "P2_1/c", "C2/c"]:
        # monoclinic
        return "red", "black"
    if label in ["P222", "P222_1", "P2_12_12", "P2_12_12_1",
                 "C222", "C222_1",
                 "F222",
                 "I222", "I2_12_12_1",
                 "Pmm2", "Pmc2_1", "Pcc2", "Pma2", "Pca2_1", "Pnc2", "Pmn2_1", "Pba2", "Pna2_1", "Pnn2",
                 "Cmm2", "Cmc2_1", "Ccc2",
                 "Amm2", "Abm2", "Ama2", "Aba2",
                 "Fmm2", "Fdd2",
                 "Imm2", "Iba2", "Ima2", "Pmmm", "Pnnn", "Pccm", "Pban", "Pmma", "Pnna", "Pmna", "Pcca", "Pbam", "Pccn",
                 "Pbcm", "Pnnm", "Pmmn", "Pbcn", "Pbca", "Pnma",
                 "Cmmm", "Cmcm", "Cmca", "Cccm", "Cmma", "Ccca",
                 "Fmmm", "Fddd",
                 "Immm", "Ibam", "Ibcm", "Imma",
                 "Cmce", "Aea2", "Aem2", "Ccce", "Ibca", "Cmme"]:
        # orthorhombic
        return "lightblue", "black"
    if label in ["P3", "P3_1", "P3_2",
                 "R3",
                 "P-3",
                 "R-3",
                 "P312", "P3_112", "P3_212",
                 "P321", "P3_121", "P3_221",
                 "R32",
                 "P31m",
                 "P31c", "P3m1", "P3c1", "R3m", "R3c",
                 "P-31m", "P-31c", "P-3m1", "P-3c1",
                 "R-3m", "R-3c"]:
        # trigonal
        return "gold", "black"
    if label in ["P6", "P6_1", "P6_2", "P6_3", "P6_4", "P6_5",
                 "P-6",
                 "P6/m", "P6_3/m",
                 "P622", "P6_122", "P6_222", "P6_322", "P6_422", "P6_522",
                 "P6mm", "P6cc", "P6_3cm", "P6_3mc",
                 "P-6m2", "P-6c2",
                 "P-62m", "P62c",
                 "P6/mmm", "P6/mcc", "P6_3/mcm", "P6_3/mmc",
                 "P-62c"]:
        # hexagonal
        return "yellow", "black"
    if label in ["P23", "P2_13",
                 "F23",
                 "I23", "I2_13",
                 "Pm-3", "Pn-3", "Pa-3",
                 "Fm-3", "Fd-3",
                 "Im-3", "Ia-3",
                 "P432", "P4_232", "P4_332", "P4_132",
                 "F432", "F4_132",
                 "I432", "I4_132",
                 "P-43m", "P-43n",
                 "F-43m", "F-43c",
                 "I-43m", "I-43d",
                 "Pm-3m", "Pn-3n", "Pm-3n", "Pn-3m",
                 "Fm-3m", "Fm-3m", "Fd-3m", "Fd-3c",
                 "Im-3m", "Ia-3d",
                 "Fm-3c"]:
        # cubic
        return "plum", "black"
    if label in ["P4", "P4_1", "P4_2", "P4_3",
                 "I4", "I4_1",
                 "P-4",
                 "I-4",
                 "P4/m", "P4_2/m", "P4/n", "P4_2/n",
                 "I4/m", "I4_1/a",
                 "P422", "P42_12", "P4_122", "P4_12_12", "P4_222",
                 "P42_2_12", "P4_322", "P4_32_12",
                 "I422", "I42_12",
                 "P4mm", "P4bm", "P4_2cm", "P4_2nm", "P4cc",
                 "P4nc", "P4_2mc", "P4_2bc",
                 "I4mm", "I4cm", "I4_1md", "I4_1cd",
                 "P-42m", "P-42c", "P-42_1m", "P-42_1c",
                 "I-42m", "I-42d",
                 "P-4m2", "P-4c2", "P-4b2", "P-4n2",
                 "I-4m2", "I-4c2",
                 "P4/mmm", "P4/mcc", "P4/nbm", "P4/nnc",
                 "P4/mbm", "P4/mnc", "P4/nmm", "P4/ncc",
                 "P4_2/mmc", "P4_2/mcm", "P4_2/nbc", "P4_2/nnm",
                 "P4_2/mbc", "P4_2/mcm", "P4_2/nmc", "P4_2/ncm",
                 "I4/mmm", "I4/mcm", "I4_1/amd", "I4_1/acd",
                 "P4_2/mnm", "P4_22_12", "I4_122"]:
        # tetragonal
        return "deepskyblue", "black"
    return "lightgray", "black"


if __name__ == "__main__":

    embeddings = atom_vectors_from_csv("../out/cif_model_24c.spacegroup_vectors.csv")

    color_map = []
    labels = []
    label_colors = []
    X = []
    for elem, embedding in embeddings.items():
        sg = elem.replace("_sg", "")
        point_color, label_color = color_by_label(sg)
        color_map.append(point_color)
        labels.append(sg)
        label_colors.append(label_color)
        X.append(embedding)
    X = np.array(X)

    technique = TSNE(n_components=2, verbose=1, perplexity=50, n_iter=500, learning_rate=10, metric="cosine")
    result = technique.fit_transform(X)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(result[:, 0], result[:, 1], c=color_map, s=[190] * len(X))
    for i, res in enumerate(result):
        ax.annotate(labels[i], (res[0], res[1]), ha="center", va="center", color=label_colors[i], fontsize=7)
    ax.set_xlabel("Dimension 1")
    ax.set_ylabel("Dimension 2")

    legend_elements = [
        Line2D([0], [0], marker="o", color="w", label="Triclinic", markerfacecolor="pink", markersize=9),
        Line2D([0], [0], marker="o", color="w", label="Monoclinic", markerfacecolor="red", markersize=9),
        Line2D([0], [0], marker="o", color="w", label="Orthorhombic", markerfacecolor="lightblue", markersize=9),
        Line2D([0], [0], marker="o", color="w", label="Tetragonal", markerfacecolor="deepskyblue", markersize=9),
        Line2D([0], [0], marker="o", color="w", label="Trigonal", markerfacecolor="gold", markersize=9),
        Line2D([0], [0], marker="o", color="w", label="Hexagonal", markerfacecolor="yellow", markersize=9),
        Line2D([0], [0], marker="o", color="w", label="Cubic", markerfacecolor="plum", markersize=9),
    ]
    ax.legend(handles=legend_elements, loc="upper left", prop={"size": 8})

    plt.show()
