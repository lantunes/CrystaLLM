import os
import torch
import tarfile
import pickle
from pymatgen.core import Element

from model import GPTConfig, GPT

from crystallm import get_cif_tokenizer


if __name__ == '__main__':
    # model_dir = "../out/cif_model_24"
    # dataset_fname = "../out/mp_oqmd_nomad_cifs_semisymm_Z_props__old.tar.gz"
    # out_fname = "../out/cif_model_24.atom_vectors.csv"
    # symmetrized, includes_props = True, True

    model_dir = "../out/cif_model_24c"
    dataset_fname = "../out/mp_oqmd_nomad_cifs_semisymm_Z_props.tar.gz"
    out_fname = "../out/cif_model_24c.atom_vectors.csv"
    symmetrized, includes_props = True, True

    tokenizer = get_cif_tokenizer(symmetrized=symmetrized, includes_props=includes_props)

    with tarfile.open(dataset_fname, "r:gz") as file:
        file_content_byte = file.extractfile("mp_oqmd_nomad_cifs_semisymm_Z_props/meta.pkl").read()
        meta = pickle.loads(file_content_byte)

    device = "cpu"
    checkpoint = torch.load(os.path.join(model_dir, "ckpt.pt"), map_location=device)
    gptconf = GPTConfig(**checkpoint["model_args"])
    model = GPT(gptconf)
    state_dict = checkpoint["model"]
    unwanted_prefix = "_orig_mod."
    for k, v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
    model.to(device)

    embedding_weights = model.transformer.wte.weight

    print(f"embedding weights shape: {embedding_weights.shape}")

    assert meta["vocab_size"] == embedding_weights.shape[0], \
        "the size of the vocab does not match the size of the embedding table"

    stoi = meta["stoi"]

    sorted_elems = sorted([(e, Element(e).number) for e in  tokenizer.atoms()], key=lambda v: v[1])
    dim = embedding_weights.shape[1]

    with open(out_fname, "wt") as f:
        header = ["element"]
        header.extend([str(i) for i in range(dim)])
        f.write("%s\n" % ",".join(header))
        for elem, _ in sorted_elems:
            vec = embedding_weights[stoi[elem]].tolist()
            row = [elem]
            row.extend([str(v) for v in vec])
            f.write("%s\n" % ",".join(row))
