from lib import atom_vectors_from_csv
import numpy as np
import matplotlib.pyplot as plt
from pymatgen.core import Element


if __name__ == '__main__':
    embeddings = atom_vectors_from_csv("../out/cif_model_24.atom_vectors.csv")
    title = "cif_model_24 atom vector similarities"

    # embeddings = atom_vectors_from_csv("../out/cif_model_24c.atom_vectors.csv")
    # title = "cif_model_24c atom vector similarities"

    elements = sorted([Element(atom) for atom in embeddings], key=lambda e: (e.group, e.row))

    similarity_matrix = [[0. for _ in range(len(elements))] for _ in range(len(elements))]

    for i, elem_i in enumerate(elements):
        for j, elem_j in enumerate(elements):
            atom_i_repr = embeddings[elem_i.name]
            atom_j_repr = embeddings[elem_j.name]

            # cosine similarity
            similarity = np.dot(atom_i_repr, atom_j_repr) / (np.linalg.norm(atom_i_repr) * np.linalg.norm(atom_j_repr))

            # euclidean distance
            # similarity = np.linalg.norm(np.array(atom_i_repr) - np.array(atom_j_repr))

            similarity_matrix[i][j] = similarity

    plt.imshow(similarity_matrix, cmap='Blues_r')
    plt.title(title)
    plt.yticks(ticks=list(range(len(similarity_matrix))), labels=[e.name for e in elements], fontsize=7)
    plt.gca().xaxis.tick_top()
    plt.xticks(ticks=list(range(len(similarity_matrix))), labels=[e.name for e in elements], fontsize=7)
    [l.set_visible(False) for (i, l) in enumerate(plt.gca().xaxis.get_ticklabels()) if i % 3 != 0]
    [l.set_visible(False) for (i, l) in enumerate(plt.gca().yaxis.get_ticklabels()) if i % 3 != 0]
    plt.colorbar()
    plt.clim(-1, 1)
    plt.show()