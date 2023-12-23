import gzip
import pickle
from sklearn.model_selection import train_test_split


if __name__ == '__main__':

    cifs_fname = "../out/orig_cifs_mp_2022_04_12+oqmd_v1_5+nomad_2023_04_30__comp-sg_augm.pkl.gz"
    train_fname = "../out/orig_cifs_mp_2022_04_12+oqmd_v1_5+nomad_2023_04_30__comp-sg_augm.train.pkl.gz"
    val_fname = "../out/orig_cifs_mp_2022_04_12+oqmd_v1_5+nomad_2023_04_30__comp-sg_augm.val.pkl.gz"
    test_fname = "../out/orig_cifs_mp_2022_04_12+oqmd_v1_5+nomad_2023_04_30__comp-sg_augm.test.pkl.gz"
    random_state = 20230610
    validation_size = 0.0155
    test_size = 0.0045

    with gzip.open(cifs_fname, "rb") as f:
        cifs = pickle.load(f)

    print("splitting dataset...")

    cifs_train, cifs_test = train_test_split(cifs, test_size=test_size,
                                             shuffle=True, random_state=random_state)

    cifs_train, cifs_val = train_test_split(cifs_train, test_size=validation_size,
                                            shuffle=True, random_state=random_state)

    print(f"number of CIFs in train set: {len(cifs_train)}")
    print(f"number of CIFs in validation set: {len(cifs_val)}")
    print(f"number of CIFs in test set: {len(cifs_test)}")

    print("writing train set...")
    with gzip.open(train_fname, "wb") as f:
        pickle.dump(cifs_train, f, protocol=pickle.HIGHEST_PROTOCOL)

    print("writing validation set...")
    with gzip.open(val_fname, "wb") as f:
        pickle.dump(cifs_val, f, protocol=pickle.HIGHEST_PROTOCOL)

    print("writing test set...")
    with gzip.open(test_fname, "wb") as f:
        pickle.dump(cifs_test, f, protocol=pickle.HIGHEST_PROTOCOL)
