import pandas as pd


def atom_vectors_from_csv(embedding_csv):
    df = pd.read_csv(embedding_csv)
    elements = list(df["element"])
    df.drop(["element"], axis=1, inplace=True)
    embeds_array = df.to_numpy()
    embedding_data = {
        elements[i]: embeds_array[i] for i in range(len(embeds_array))
    }
    return embedding_data