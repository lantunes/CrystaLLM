from lib import atom_vectors_from_csv
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import numpy as np


if __name__ == '__main__':

    embeddings = atom_vectors_from_csv("../out/cif_model_35.number_vectors.csv")
    out_fname = "../out/digit_vectors.pdf"

    labels = []
    X = []
    for elem, embedding in embeddings.items():
        labels.append(elem)
        X.append(embedding)
    X = np.array(X)

    technique = PCA(n_components=2)
    result = technique.fit_transform(X)

    pc1_var, pc2_var = technique.explained_variance_ratio_

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(result[:, 0], result[:, 1], s=[190]*len(X))
    for i, res in enumerate(result):
        ax.annotate(labels[i], (res[0], res[1]), ha="center", va="center", color="w")
    ax.set_xlabel(f"PC1 ({pc1_var*100:.2f}%)")
    ax.set_ylabel(f"PC2 ({pc2_var*100:.2f}%)")

    plt.savefig(out_fname)
    plt.show()
