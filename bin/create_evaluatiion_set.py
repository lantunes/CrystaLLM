import numpy as np
from lib import get_cif_tokenizer, CIFData, populate_cif_data
from tqdm import tqdm
import csv


"""
We construct the evaluation set from the CIFs in the validation set,
since the model didn't directly see the CIFs of the validation set in 
training. 
"""
if __name__ == '__main__':
    validation_set_fname = "../out/mp_oqmd_cifs_nosymm/val.bin"
    out_file = "../out/mp_oqmd_cifs_nosymm/eval.csv"
    n = 10_000  # the number of CIFs to randomly select to include in the evaluation set
    symmetrized = False

    validation_set_ints = np.fromfile(validation_set_fname, dtype=np.uint16)
    tokenizer = get_cif_tokenizer(symmetrized)

    # validation_set_tokens = tokenizer.decode(validation_set_ints)
    id_to_token = tokenizer.id_to_token

    cif_datas = []

    curr_cif_data = None
    curr_cif_tokens = []

    for i in tqdm(validation_set_ints):
        token = id_to_token[i]

        if token == "data_":
            # populate the existing CIFData if it exists
            if curr_cif_data is not None:
                populate_cif_data(curr_cif_data, ''.join(curr_cif_tokens))
                cif_datas.append(curr_cif_data)

            # start a new CIFData
            curr_cif_data = CIFData()
            curr_cif_tokens = []

        curr_cif_tokens.append(token)
    # we may have a CIFData that's in progress, but we won't add it to cif_datas, since we
    #  don't know if it's complete (i.e. all the necessary tokens have been read)

    selected_i = np.random.choice(range(len(cif_datas)), n)

    with open(out_file, "wt") as f:
        writer = csv.writer(f)
        writer.writerow([
            "composition",
            "cell_length_a",
            "cell_length_b",
            "cell_length_c",
            "cell_angle_alpha",
            "cell_angle_beta",
            "cell_angle_gamma",
            "cell_volume"
        ])
        for i in selected_i:
            writer.writerow([
                cif_datas[i].composition,
                cif_datas[i].cell_length_a,
                cif_datas[i].cell_length_b,
                cif_datas[i].cell_length_c,
                cif_datas[i].cell_angle_alpha,
                cif_datas[i].cell_angle_beta,
                cif_datas[i].cell_angle_gamma,
                cif_datas[i].cell_volume
            ])
