from lib import atom_vectors_from_csv
import numpy as np
import matplotlib.pyplot as plt


if __name__ == '__main__':
    embeddings = atom_vectors_from_csv("../out/cif_model_24c.number_vectors.csv")
    title = "cif_model_24c number vector similarities"

    elements = list(range(10))

    similarity_matrix = [[0. for _ in range(len(elements))] for _ in range(len(elements))]

    for i, elem_i in enumerate(elements):
        for j, elem_j in enumerate(elements):
            num_i_repr = embeddings[elem_i]
            num_j_repr = embeddings[elem_j]

            # cosine similarity
            similarity = np.dot(num_i_repr, num_j_repr) / (np.linalg.norm(num_i_repr) * np.linalg.norm(num_j_repr))

            # euclidean distance
            # similarity = np.linalg.norm(np.array(num_i_repr) - np.array(num_j_repr))

            similarity_matrix[i][j] = similarity

    plt.imshow(similarity_matrix, cmap='Blues_r')
    plt.title(title)
    plt.yticks(ticks=list(range(len(similarity_matrix))), labels=elements, fontsize=7)
    plt.gca().xaxis.tick_top()
    plt.xticks(ticks=list(range(len(similarity_matrix))), labels=elements, fontsize=7)
    [l.set_visible(False) for (i, l) in enumerate(plt.gca().xaxis.get_ticklabels()) if i % 3 != 0]
    [l.set_visible(False) for (i, l) in enumerate(plt.gca().yaxis.get_ticklabels()) if i % 3 != 0]
    plt.colorbar()
    plt.clim(-1, 1)
    plt.show()