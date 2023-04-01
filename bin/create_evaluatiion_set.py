import numpy as np
from lib import get_cif_tokenizer
from tqdm import tqdm
import csv


class CIFData:
    def __init__(self):
        self.composition = None
        self.cell_length_a = None
        self.cell_length_b = None
        self.cell_length_c = None
        self.cell_angle_alpha = None
        self.cell_angle_beta = None
        self.cell_angle_gamma = None
        self.cell_volume = None

    def __repr__(self):
        return f"CIFData[" \
               f"comp:'{self.composition}', " \
               f"a:{self.cell_length_a}, " \
               f"b:{self.cell_length_b}, " \
               f"c:{self.cell_length_c}, " \
               f"alpha:{self.cell_angle_alpha}, " \
               f"beta:{self.cell_angle_beta}, " \
               f"gamma:{self.cell_angle_gamma}, " \
               f"vol:{self.cell_volume}, " \
               f"]"

    def to_csv_row(self):
        return [
            self.composition,
            self.cell_length_a,
            self.cell_length_b,
            self.cell_length_c,
            self.cell_angle_alpha,
            self.cell_angle_beta,
            self.cell_angle_gamma,
            self.cell_volume
        ]


def populate_cif_data(cif_data, cif):
    for line in cif.split("\n"):
        if line.startswith("data_"):
            cif_data.composition = line.split("_")[1]
        elif line.startswith("_cell_length_a"):
            cif_data.cell_length_a = float(line.split(" ")[1].strip())
        elif line.startswith("_cell_length_b"):
            cif_data.cell_length_b = float(line.split(" ")[1].strip())
        elif line.startswith("_cell_length_c"):
            cif_data.cell_length_c = float(line.split(" ")[1].strip())
        elif line.startswith("_cell_angle_alpha"):
            cif_data.cell_angle_alpha = float(line.split(" ")[1].strip())
        elif line.startswith("_cell_angle_beta"):
            cif_data.cell_angle_beta = float(line.split(" ")[1].strip())
        elif line.startswith("_cell_angle_gamma"):
            cif_data.cell_angle_gamma = float(line.split(" ")[1].strip())
        elif line.startswith("_cell_volume"):
            cif_data.cell_volume = float(line.split(" ")[1].strip())

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
