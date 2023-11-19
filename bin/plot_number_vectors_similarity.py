from lib import atom_vectors_from_csv
import matplotlib.pyplot as plt
import matplotlib.pylab as pl
import numpy as np


def cosine_similarity(num_i_repr, num_j_repr):
    return np.dot(num_i_repr, num_j_repr) / (np.linalg.norm(num_i_repr) * np.linalg.norm(num_j_repr))


def euclidean_distance(num_i_repr, num_j_repr):
    return np.linalg.norm(np.array(num_i_repr) - np.array(num_j_repr))


if __name__ == '__main__':

    embeddings = atom_vectors_from_csv("../out/cif_model_35.number_vectors.csv")
    # sim_type, out_fname = "Cosine", "../out/digit_vectors_similarities_cosine.pdf"
    sim_type, out_fname = "Euclidean", "../out/digit_vectors_similarities_euclid.pdf"

    colors = pl.cm.tab10(np.linspace(0, 1, 10))

    for digit1 in list(range(10)):
        vec1 = embeddings[digit1]
        X = []
        Y = []
        for digit2, vec2 in embeddings.items():
            X.append(digit2)
            if sim_type == "Euclidean":
                Y.append(euclidean_distance(vec1, vec2))
            elif sim_type == "Cosine":
                Y.append(cosine_similarity(vec1, vec2))
            else:
                raise Exception(f"unsupported similarity type: {sim_type}")

        alpha = 1 if digit1 == 5 else 0.3
        plt.plot(X, Y, label=digit1, alpha=alpha, color=colors[digit1])

    plt.xticks(list(range(10)))
    plt.xlabel("Digit")
    plt.ylabel(f"{sim_type} distance")
    plt.title(f"{sim_type} distances between each digit")
    plt.legend()

    plt.savefig(out_fname)
    plt.show()
