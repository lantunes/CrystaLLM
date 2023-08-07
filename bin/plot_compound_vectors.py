import numpy as np
import matplotlib.pyplot as plt
from lib import atom_vectors_from_csv
from sklearn.decomposition import PCA


if __name__ == '__main__':

    # embeddings = atom_vectors_from_csv("../out/cif_model_24.atom_vectors.csv")
    embeddings = atom_vectors_from_csv("../out/cif_model_24c.atom_vectors.csv")

    pool = np.mean

    O_embedding = embeddings["O"]
    Ni_embedding = embeddings["Ni"]
    Cr_embedding = embeddings["Cr"]
    Zr_embedding = embeddings["Zr"]

    colors = []
    labels = []
    X = []

    labels.append("$Zr$")
    colors.append("green")
    X.append(Zr_embedding)
    labels.append("$Cr$")
    colors.append("red")
    X.append(Cr_embedding)
    labels.append("$Ni$")
    colors.append("blue")
    X.append(Ni_embedding)
    labels.append("$NiO$")
    colors.append("blue")
    X.append(pool([Ni_embedding, O_embedding], axis=0))
    labels.append("$ZrO_2$")
    colors.append("green")
    X.append(pool([Zr_embedding, O_embedding, O_embedding], axis=0))
    labels.append("$Cr_2O_3$")
    colors.append("red")
    X.append(pool([Cr_embedding, Cr_embedding, O_embedding, O_embedding, O_embedding], axis=0))

    pca = PCA(n_components=2)
    result = pca.fit_transform(X)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(result[:, 0], result[:, 1], c=colors, s=[80]*6)
    for i, res in enumerate(result):
        ax.annotate(labels[i], (res[0], res[1]), xytext=(res[0]+0.075, res[1]))
    ax.set_xlabel("First principal component")
    ax.set_ylabel("Second principal component")
    ax.set_xlim([-2.6, 2.6])
    plt.show()