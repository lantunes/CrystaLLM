import tarfile
import pickle
import numpy as np
from tqdm import tqdm
from lib import extract_data_formula, extract_space_group_symbol


def get_underrepresented_set(underrepresented_fname):
    with open(underrepresented_fname, "rb") as f:
        comps = pickle.load(f)
    underrepresented_set = set()
    for comp, sg in comps:
        underrepresented_set.add(f"{comp}_{sg}")
    return underrepresented_set


if __name__ == '__main__':
    dataset_fname = "../out/mp_oqmd_nomad_cifs_semisymm_Z_props_all.tar.gz"
    # underrepresented_fname = "../out/underrepresented.pkl"
    underrepresented_fname = "../out/underrep_lengths_gt500_lte2048.pkl"
    all_cif_start_indices_out_fname = "../out/mp_oqmd_nomad_cifs_semisymm_Z_props_all.starts.pkl"
    # underrepresented_cif_start_indices_out_fname = \
    #     "../out/mp_oqmd_nomad_cifs_semisymm_Z_props_all.starts_underrep.pkl"
    underrepresented_cif_start_indices_out_fname = \
        "../out/mp_oqmd_nomad_cifs_semisymm_Z_props_all.starts_underrep_lengths_gt500_lte2048.pkl"

    with tarfile.open(dataset_fname, "r:gz") as file:
        file_content_byte = file.extractfile("mp_oqmd_nomad_cifs_semisymm_Z_props_all/meta.pkl").read()
        meta = pickle.loads(file_content_byte)

        extracted = file.extractfile("mp_oqmd_nomad_cifs_semisymm_Z_props_all/train.bin")
        train_ids = np.frombuffer(extracted.read(), dtype=np.uint16)

    underrepresented_set = get_underrepresented_set(underrepresented_fname)

    all_cif_start_indices = []
    underrepresented_start_indices = []

    curr_cif_tokens = []

    for i, id in tqdm(enumerate(train_ids), total=len(train_ids)):
        token = meta["itos"][id]

        if token == "data_":
            all_cif_start_indices.append(i)

        curr_cif_tokens.append(token)

        if len(curr_cif_tokens) > 1 and curr_cif_tokens[-2:] == ['\n', '\n']:
            # we're at the end of the CIF;
            #  reconstruct the CIF and see if it's in the underrepresented set
            cif = ''.join(curr_cif_tokens)
            data_formula = extract_data_formula(cif)
            space_group_symbol = extract_space_group_symbol(cif)

            if f"{data_formula}_{space_group_symbol}" in underrepresented_set:
                # the last added start index is the start index of this CIF
                underrepresented_start_indices.append(all_cif_start_indices[-1])

            curr_cif_tokens = []

    with open(all_cif_start_indices_out_fname, "wb") as f:
        pickle.dump(all_cif_start_indices, f, protocol=pickle.HIGHEST_PROTOCOL)

    with open(underrepresented_cif_start_indices_out_fname, "wb") as f:
        pickle.dump(underrepresented_start_indices, f, protocol=pickle.HIGHEST_PROTOCOL)
