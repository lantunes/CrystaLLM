import os
import argparse
from collections import Counter
import tarfile

import numpy as np
import itertools
from tqdm import tqdm

import smact
from smact.screening import pauling_test
from scipy.stats import wasserstein_distance
from scipy.spatial.distance import cdist

from pymatgen.core import Structure, Composition
from pymatgen.analysis.structure_matcher import StructureMatcher
from matminer.featurizers.site.fingerprint import CrystalNNFingerprint
from matminer.featurizers.composition.composite import ElementProperty

from crystallm import is_sensible, extract_data_formula

import warnings
warnings.filterwarnings("ignore")

THIS_DIR = os.path.dirname(os.path.abspath(__file__))

CrystalNNFP = CrystalNNFingerprint.from_preset("ops")
CompFP = ElementProperty.from_preset("magpie")

COV_Cutoffs = {
    "mp20": {"struc": 0.4, "comp": 10.},
    "carbon": {"struc": 0.2, "comp": 4.},
    "perovskite": {"struc": 0.2, "comp": 4},
}


class StandardScaler:
    def __init__(self, means, stds):
        self.means = means
        self.stds = stds

    def transform(self, X):
        X = np.array(X).astype(float)
        return (X - self.means) / self.stds


# adapted from
#  https://github.com/jiaor17/DiffCSP/blob/ee131b03a1c6211828e8054d837caa8f1a980c3e/scripts/eval_utils.py
def smact_validity(atom_types, use_pauling_test=True, include_alloys=True):
    # atom_types e.g. ["Fe", "Fe", "O", "O", "O"]
    elem_counter = Counter(atom_types)
    elems = [(elem, elem_counter[elem]) for elem in sorted(elem_counter.keys())]
    comp, elem_counts = list(zip(*elems))
    elem_counts = np.array(elem_counts)
    elem_counts = elem_counts / np.gcd.reduce(elem_counts)
    count = tuple(elem_counts.astype("int").tolist())

    elem_symbols = tuple(comp)
    space = smact.element_dictionary(elem_symbols)
    smact_elems = [e[1] for e in space.items()]
    electronegs = [e.pauling_eneg for e in smact_elems]
    ox_combos = [e.oxidation_states for e in smact_elems]
    if len(set(elem_symbols)) == 1:
        return True
    if include_alloys:
        is_metal_list = [elem_s in smact.metals for elem_s in elem_symbols]
        if all(is_metal_list):
            return True
    threshold = np.max(count)
    oxn = 1
    for oxc in ox_combos:
        oxn *= len(oxc)
    if oxn > 1e7:
        return False
    for ox_states in itertools.product(*ox_combos):
        stoichs = [(c,) for c in count]
        # Test for charge balance
        cn_e, cn_r = smact.neutral_ratios(
            ox_states, stoichs=stoichs, threshold=threshold)
        # Electronegativity test
        if cn_e:
            if use_pauling_test:
                try:
                    electroneg_OK = pauling_test(ox_states, electronegs)
                except TypeError:
                    # if no electronegativity data, assume it is okay
                    electroneg_OK = True
            else:
                electroneg_OK = True
            if electroneg_OK:
                return True
    return False


def get_comp_fingerprint(struct):
    atom_types = [str(specie) for specie in struct.species]
    elem_counter = Counter(atom_types)
    comp = Composition(elem_counter)
    fp = CompFP.featurize(comp)
    if np.isnan(fp).any():
        return None
    return fp


def get_struct_fingerprint(struct):
    try:
        site_fps = [CrystalNNFP.featurize(struct, i) for i in range(len(struct))]
    except Exception:
        return None
    return np.array(site_fps).mean(axis=0)


# from https://github.com/jiaor17/DiffCSP/blob/ee131b03a1c6211828e8054d837caa8f1a980c3e/scripts/eval_utils.py
def structure_validity(crystal, cutoff=0.5):
    dist_mat = crystal.distance_matrix
    # Pad diagonal with a large number
    dist_mat = dist_mat + np.diag(
        np.ones(dist_mat.shape[0]) * (cutoff + 10.))
    if dist_mat.min() < cutoff or crystal.volume < 0.1:
        return False
    else:
        return True


def is_valid(struct):
    comp_valid = smact_validity(
        atom_types=[str(specie) for specie in struct.species]
    )
    struct_valid = structure_validity(struct)
    return comp_valid and struct_valid


def is_valid_unconditional(struct, fp):
    return is_valid(struct) and fp is not None


# from https://github.com/jiaor17/DiffCSP/blob/ee131b03a1c6211828e8054d837caa8f1a980c3e/scripts/eval_utils.py
def filter_fps(struc_fps, comp_fps):
    assert len(struc_fps) == len(comp_fps)

    filtered_struc_fps, filtered_comp_fps = [], []

    for struc_fp, comp_fp in zip(struc_fps, comp_fps):
        if struc_fp is not None and comp_fp is not None:
            filtered_struc_fps.append(struc_fp)
            filtered_comp_fps.append(comp_fp)
    return filtered_struc_fps, filtered_comp_fps


# adapted from
#  https://github.com/jiaor17/DiffCSP/blob/ee131b03a1c6211828e8054d837caa8f1a980c3e/scripts/eval_utils.py
def compute_cov(gen_structs, true_structs, struc_cutoff, comp_cutoff, comp_scaler, num_gen_crystals=None):
    struc_fps = [struct_fp for _, struct_fp, _ in gen_structs]
    comp_fps = [comp_fp for _, _, comp_fp in gen_structs]
    gt_struc_fps = [struct_fp for _, struct_fp, _ in true_structs]
    gt_comp_fps = [comp_fp for _, _, comp_fp in true_structs]

    assert len(struc_fps) == len(comp_fps)
    assert len(gt_struc_fps) == len(gt_comp_fps)

    # Use number of crystal before filtering to compute COV
    if num_gen_crystals is None:
        num_gen_crystals = len(struc_fps)

    struc_fps, comp_fps = filter_fps(struc_fps, comp_fps)
    # there may be odd cases when ground-truth CIFs may result in
    #  fingerprints with nan values; in those cases, we return None
    #  instead of the fingerprint, and consolidate those entries here
    gt_struc_fps, gt_comp_fps = filter_fps(gt_struc_fps, gt_comp_fps)

    comp_fps = comp_scaler.transform(comp_fps)
    gt_comp_fps = comp_scaler.transform(gt_comp_fps)

    struc_fps = np.array(struc_fps)
    gt_struc_fps = np.array(gt_struc_fps)
    comp_fps = np.array(comp_fps)
    gt_comp_fps = np.array(gt_comp_fps)

    struc_pdist = cdist(struc_fps, gt_struc_fps)
    comp_pdist = cdist(comp_fps, gt_comp_fps)

    struc_recall_dist = struc_pdist.min(axis=0)
    struc_precision_dist = struc_pdist.min(axis=1)
    comp_recall_dist = comp_pdist.min(axis=0)
    comp_precision_dist = comp_pdist.min(axis=1)

    cov_recall = np.mean(np.logical_and(
        struc_recall_dist <= struc_cutoff,
        comp_recall_dist <= comp_cutoff))
    cov_precision = np.sum(np.logical_and(
        struc_precision_dist <= struc_cutoff,
        comp_precision_dist <= comp_cutoff)) / num_gen_crystals

    metrics_dict = {
        "cov_recall": cov_recall,
        "cov_precision": cov_precision,
        "amsd_recall": np.mean(struc_recall_dist),
        "amsd_precision": np.mean(struc_precision_dist),
        "amcd_recall": np.mean(comp_recall_dist),
        "amcd_precision": np.mean(comp_precision_dist),
    }

    combined_dist_dict = {
        "struc_recall_dist": struc_recall_dist.tolist(),
        "struc_precision_dist": struc_precision_dist.tolist(),
        "comp_recall_dist": comp_recall_dist.tolist(),
        "comp_precision_dist": comp_precision_dist.tolist(),
    }

    return metrics_dict, combined_dist_dict


# adapted from
#  https://github.com/jiaor17/DiffCSP/blob/ee131b03a1c6211828e8054d837caa8f1a980c3e/scripts/compute_metrics.py
def get_match_rate_and_rms(gen_structs, true_structs, matcher):
    def process_one(pred, gt, is_pred_valid):
        if not is_pred_valid:
            return None
        try:
            rms_dist = matcher.get_rms_dist(pred, gt)
            rms_dist = None if rms_dist is None else rms_dist[0]
            return rms_dist
        except Exception:
            return None

    rms_dists = []
    for i in tqdm(range(len(gen_structs)), desc="comparing structures..."):
        tmp_rms_dists = []
        for j in range(len(gen_structs[i])):
            try:
                struct_valid = is_valid(gen_structs[i][j])
                rmsd = process_one(gen_structs[i][j], true_structs[i], struct_valid)
                if rmsd is not None:
                    tmp_rms_dists.append(rmsd)
            except Exception:
                pass
        if len(tmp_rms_dists) == 0:
            rms_dists.append(None)
        else:
            rms_dists.append(np.min(tmp_rms_dists))

    rms_dists = np.array(rms_dists)
    match_rate = sum(rms_dists != None) / len(gen_structs)
    mean_rms_dist = rms_dists[rms_dists != None].mean()
    return {"match_rate": match_rate, "rms_dist": mean_rms_dist}


# adapted from
#  https://github.com/jiaor17/DiffCSP/blob/ee131b03a1c6211828e8054d837caa8f1a980c3e/scripts/compute_metrics.py
def get_unconditional_metrics(gen_structs, gen_comps, true_structs, n_gen, comp_scaler, cov_cutoffs, n_samples=1000):
    valid_structs = []
    for struct, struct_fp, _ in tqdm(gen_structs, desc="getting valid structures..."):
        if is_valid_unconditional(struct, struct_fp):
            valid_structs.append(struct)
    if len(valid_structs) >= n_samples:
        sampled_indices = np.random.choice(len(valid_structs), n_samples, replace=False)
        valid_samples = [valid_structs[i] for i in sampled_indices]
    else:
        raise Exception(
            f"Insufficient valid crystals in the generated set: {len(valid_structs)}/{n_samples}")

    n_comp_valid = 0
    for comp in tqdm(gen_comps, desc="counting comp valid..."):
        # even if a structure is unreasonable or invalid,
        #  the generated composition might still be valid
        try:
            if smact_validity(
                atom_types=[str(elem) for elem, n in comp.items() for _ in range(int(n))]
            ):
                n_comp_valid += 1
        except Exception:
            pass
    comp_valid = n_comp_valid / n_gen
    n_struct_valid = 0
    for struct, _, _ in tqdm(gen_structs, desc="counting struct valid..."):
        if structure_validity(struct):
            n_struct_valid += 1
    struct_valid = n_struct_valid / n_gen
    valid = len(valid_structs) / n_gen
    valid_dict = {"comp_valid": comp_valid, "struct_valid": struct_valid, "valid": valid}

    print("computing wdist_density...")
    pred_densities = [struct.density for struct in valid_samples]
    gt_densities = [struct.density for struct, _, _ in true_structs]
    wdist_density = wasserstein_distance(pred_densities, gt_densities)
    wdist_density_dict = {"wdist_density": wdist_density}

    print("computing wdist_num_elems...")
    pred_nelems = [len(set(struct.species)) for struct in valid_samples]
    gt_nelems = [len(set(struct.species)) for struct, _, _ in true_structs]
    wdist_num_elems = wasserstein_distance(pred_nelems, gt_nelems)
    wdist_num_elems_dict = {"wdist_num_elems": wdist_num_elems}

    # TODO use property models to compute formation energy Wasserstein distances

    print("computing cov...")
    cutoff_dict = COV_Cutoffs[cov_cutoffs]
    cov_metrics_dict, _ = compute_cov(
        gen_structs,
        true_structs,
        struc_cutoff=cutoff_dict["struc"],
        comp_cutoff=cutoff_dict["comp"],
        comp_scaler=comp_scaler,
    )

    metrics = {}
    metrics.update(valid_dict)
    metrics.update({"n_sensible": len(gen_structs)})
    metrics.update(wdist_density_dict)
    # metrics.update(wdist_prop_dict)  # TODO
    metrics.update(wdist_num_elems_dict)
    metrics.update(cov_metrics_dict)

    return metrics


def get_comp_scaler_means_stds():
    with open(os.path.join(THIS_DIR, "../resources/comp_scaler_means.txt"), "rt") as f:
        comp_scaler_means = [float(num.strip()) for num in f.readlines()]
    with open(os.path.join(THIS_DIR, "../resources/comp_scaler_stds.txt"), "rt") as f:
        comp_scaler_stds = [float(num.strip()) for num in f.readlines()]
    return comp_scaler_means, comp_scaler_stds


def extract_cif_id(filepath):
    """
    Parses a filename assumed to be in the format "id__n.cif",
    returning the "id".

    :param filepath: a filename assumed to be in the format "id__n.cif"
    :return: the extracted values of `id`
    """
    filename = os.path.basename(filepath)
    # split from the right, once
    parts = filename.rsplit("__", 1)
    if len(parts) == 2:
        id_part, _ = parts
        return id_part
    else:
        raise ValueError(f"'{filename}' does not conform to expected format 'id__n.cif'")


def read_generated_cifs(input_path):
    generated_cifs = {}
    with tarfile.open(input_path, "r:gz") as tar:
        for member in tqdm(tar.getmembers(), desc="extracting generated CIFs..."):
            f = tar.extractfile(member)
            if f is not None:
                cif = f.read().decode("utf-8")
                cif_id = extract_cif_id(member.name)
                if cif_id not in generated_cifs:
                    generated_cifs[cif_id] = []
                generated_cifs[cif_id].append(cif)
    return generated_cifs


def read_true_cifs(input_path):
    true_cifs = {}
    with tarfile.open(input_path, "r:gz") as tar:
        for member in tqdm(tar.getmembers(), desc="extracting true CIFs..."):
            f = tar.extractfile(member)
            if f is not None:
                cif = f.read().decode("utf-8")
                filename = os.path.basename(member.name)
                cif_id = filename.replace(".cif", "")
                true_cifs[cif_id] = cif
    return true_cifs


def get_structs(id_to_gen_cifs, id_to_true_cifs, n_gens, length_lo, length_hi, angle_lo, angle_hi):
    gen_structs = []
    true_structs = []
    for id, cifs in tqdm(id_to_gen_cifs.items(), desc="converting CIFs to Structures..."):
        if id not in id_to_true_cifs:
            raise Exception(f"could not find ID `{id}` in true CIFs")

        structs = []
        for cif in cifs[:n_gens]:
            try:
                if not is_sensible(cif, length_lo, length_hi, angle_lo, angle_hi):
                    continue
                structs.append(Structure.from_str(cif, fmt="cif"))
            except Exception:
                pass
        gen_structs.append(structs)

        true_structs.append(Structure.from_str(id_to_true_cifs[id], fmt="cif"))
    return gen_structs, true_structs


def get_gen_comps(id_to_gen_cifs):
    gen_comps = []
    for cifs in tqdm(id_to_gen_cifs.values(), desc="extracting generated compositions from CIFs..."):
        cif = cifs[0]
        try:
            data_formula = extract_data_formula(cif)
            comp = Composition(data_formula)
            if len(comp) == 0:
                continue
            gen_comps.append(comp)
        except Exception:
            pass
    return gen_comps


def get_gen_structs_unconditional(id_to_gen_cifs, length_lo, length_hi, angle_lo, angle_hi):
    gen_structs = []
    for cifs in tqdm(id_to_gen_cifs.values(), desc="converting CIFs to Structures and fingerprints..."):
        cif = cifs[0]
        try:
            if not is_sensible(cif, length_lo, length_hi, angle_lo, angle_hi):
                continue
            struct = Structure.from_str(cif, fmt="cif")
            # get the structure fingerprint only for a valid structure
            struct_fp = get_struct_fingerprint(struct) if structure_validity(struct) else None
            comp_fp = get_comp_fingerprint(struct)
            gen_structs.append((struct, struct_fp, comp_fp))
        except Exception:
            pass
    return gen_structs


def get_true_structs_unconditional(id_to_true_cifs):
    true_structs = []
    for cif in tqdm(id_to_true_cifs.values(), desc="converting true CIFs to Structures and fingerprints..."):
        struct = Structure.from_str(cif, fmt="cif")
        struct_fp = get_struct_fingerprint(struct)
        comp_fp = get_comp_fingerprint(struct)
        true_structs.append((struct, struct_fp, comp_fp))
    return true_structs


"""
This script performs the CDVAE and DiffCSP benchmark analysis, as described in:
https://github.com/jiaor17/DiffCSP/blob/ee131b03a1c6211828e8054d837caa8f1a980c3e/scripts/compute_metrics.py.
"""
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Perform benchmark analysis.")
    parser.add_argument("gen_cifs",
                        help="Path to the .tar.gz file containing the generated CIF files.")
    parser.add_argument("true_cifs",
                        help="Path to the .tar.gz file containing the true CIF files.")
    parser.add_argument("--num-gens", required=False, default=0, type=int,
                        help="The maximum number of generations to use per structure. Default is 0, which means "
                             "use all of the available generations. (This argument is ignored for the unconditional "
                             "generation task metrics.)")
    parser.add_argument("--length_lo", required=False, default=0.5, type=float,
                        help="The smallest cell length allowable for the sensibility check")
    parser.add_argument("--length_hi", required=False, default=1000., type=float,
                        help="The largest cell length allowable for the sensibility check")
    parser.add_argument("--angle_lo", required=False, default=10., type=float,
                        help="The smallest cell angle allowable for the sensibility check")
    parser.add_argument("--angle_hi", required=False, default=170., type=float,
                        help="The largest cell angle allowable for the sensibility check")
    parser.add_argument("--unconditional", action="store_true",
                        help="If included, the unconditional generation task metrics will be computed "
                             "instead of the CSP task metrics")
    parser.add_argument("--cov-cutoffs", choices=["mp20", "carbon", "perovskite"],
                        required=False, default="perovskite",
                        help="The coverage cutoffs to use if the unconditional generation task metrics are "
                             "being computed. Default is 'perovskite'.")
    parser.add_argument("--seed", type=int, default=1337,
                        help="The random seed to use for the unconditional generation task metrics.")
    args = parser.parse_args()

    gen_cifs_path = args.gen_cifs
    true_cifs_path = args.true_cifs
    n_gens = args.num_gens
    length_lo = args.length_lo
    length_hi = args.length_hi
    angle_lo = args.angle_lo
    angle_hi = args.angle_hi
    unconditional = args.unconditional
    cov_cutoffs = args.cov_cutoffs
    seed = args.seed

    if n_gens == 0:
        n_gens = None
        print("using all available generations...")
    else:
        if n_gens < 0:
            raise Exception(f"invalid value for n_gens: {n_gens}")
        print(f"using a maximum of {n_gens} generation(s) per compound...")

    # defaults taken from DiffCSP
    struct_matcher = StructureMatcher(stol=0.5, angle_tol=10, ltol=0.3)

    id_to_gen_cifs = read_generated_cifs(gen_cifs_path)
    id_to_true_cifs = read_true_cifs(true_cifs_path)

    if unconditional:
        np.random.seed(seed)
        comp_scaler_means, comp_scaler_stds = get_comp_scaler_means_stds()
        comp_scaler = StandardScaler(
            means=np.array(comp_scaler_means),
            stds=np.array(comp_scaler_stds),
        )
        gen_structs = get_gen_structs_unconditional(
            id_to_gen_cifs, length_lo, length_hi, angle_lo, angle_hi
        )
        gen_comps = get_gen_comps(id_to_gen_cifs)
        true_structs = get_true_structs_unconditional(id_to_true_cifs)
        n_gens = len(id_to_gen_cifs)
        metrics = get_unconditional_metrics(gen_structs, gen_comps, true_structs, n_gens, comp_scaler, cov_cutoffs)
    else:
        gen_structs, true_structs = get_structs(
            id_to_gen_cifs, id_to_true_cifs, n_gens, length_lo, length_hi, angle_lo, angle_hi
        )
        metrics = get_match_rate_and_rms(gen_structs, true_structs, struct_matcher)

    print(metrics)
