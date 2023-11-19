from lib import atom_vectors_from_csv
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import numpy as np


def color_by_label(label):
    if label in ["H", "Li", "Na", "K", "Rb", "Cs"]:
        return "red", "white"
    if label in ["Be", "Mg", "Ca", "Sr", "Ba"]:
        return "brown", "white"
    if label in ["B", "Al", "Ga", "In", "Tl"]:
        return "steelblue", "white"
    if label in ["C", "Si", "Ge", "Sn", "Pb"]:
        return "purple", "white"
    if label in ["N", "P", "As", "Sb", "Bi"]:
        return "blue", "white"
    if label in ["O", "S", "Se", "Te"]:
        return "yellow", "black"
    if label in ["F", "Cl", "Br", "I"]:
        return "cyan", "black"
    if label in ["He", "Ne", "Ar", "Kr", "Xe"]:
        return "olive", "white"
    if label in ["Ni", "Pd", "Pt", "Cu", "Ag", "Au", "Zn", "Cd", "Hg", "Rh", "Ir"]:
        return "orange", "white"
    if label in ["Ti", "Zr", "Hf", "V", "Nb", "Ta", "Cr", "Mo", "W"]:
        return "green", "white"
    if label in ["Fe", "Mn", "Tc", "Ru", "Re", "Os", "Co"]:
        return "lime", "black"
    return "lightgray", "black"


if __name__ == '__main__':

    # plot_technique = "pca"
    plot_technique = "tsne"

    embeddings = atom_vectors_from_csv("../out/cif_model_35.atom_vectors.csv")
    out_fname = "../out/atom_embeddings.pdf"

    color_map = []
    labels = []
    label_colors = []
    X = []
    for atom, embedding in embeddings.items():
        point_color, label_color = color_by_label(atom)
        color_map.append(point_color)
        labels.append(atom)
        label_colors.append(label_color)
        X.append(embedding)
    X = np.array(X)

    if plot_technique == "pca":
        technique = PCA(n_components=2)
    elif plot_technique == "tsne":
        technique = TSNE(n_components=2, verbose=1, perplexity=50, n_iter=500, learning_rate=10, metric="cosine")
    else:
        raise Exception(f"unknown technique: {plot_technique}")
    result = technique.fit_transform(X)

    if plot_technique == "pca":
        print(technique.explained_variance_ratio_)

    plt.rcParams["figure.figsize"] = (12, 7)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(result[:, 0], result[:, 1], c=color_map, s=[190]*len(X))
    for i, res in enumerate(result):
        ax.annotate(labels[i], (res[0], res[1]), ha="center", va="center", color=label_colors[i])
    ax.set_xlabel("Dimension 1")
    ax.set_ylabel("Dimension 2")

    plt.savefig(out_fname)
    plt.show()
