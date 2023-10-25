import os.path
import shutil
import csv
import pandas as pd
from zipfile import ZipFile
from pymatgen.core import Structure
from pymatgen.analysis.structure_matcher import StructureMatcher, ElementComparator
from pymatgen.analysis.chemenv.coordination_environments.chemenv_strategies import SimplestChemenvStrategy, \
    AbstractChemenvStrategy
from pymatgen.analysis.chemenv.coordination_environments.coordination_geometries import AllCoordinationGeometries
from pymatgen.analysis.chemenv.coordination_environments.coordination_geometry_finder import LocalGeometryFinder
from pymatgen.analysis.chemenv.utils.chemenv_errors import NeighborsNotComputedChemenvError

import warnings
warnings.filterwarnings("ignore")


def get_formulas(model, model_dir, sg):
    formulas = []
    mcts_path = os.path.join(model_dir, f"{model}_mcts{'_sg' if sg else ''}")
    items = os.listdir(mcts_path)
    for item in items:
        if not os.path.isfile(os.path.join(mcts_path, item)):
            formulas.append(item)
    return formulas


def read_challenge_set(challenge_set_path):
    input_zip = ZipFile(challenge_set_path)
    challenge_set = {}

    for zipfile in input_zip.filelist:
        components = zipfile.filename.split("/")

        if len(components) < 3 or len(components[-1]) == 0:
            continue

        formula = components[1]
        fname = components[2]
        if fname.endswith("pymatgen.cif"):
            content = input_zip.read(zipfile.filename).decode("utf-8")
            challenge_set[formula] = content

    return challenge_set


def read_alignn_energies(fname):
    energies = {}
    with open(fname, "rt") as f:
        reader = csv.reader(f)
        next(reader)  # skip header
        for line in reader:
            formula = line[0]
            energy = float(line[1])
            energies[formula] = energy
    return energies


def read_cif(model_path, formula, fname):
    cif_path = os.path.join(model_path, formula, fname)
    with open(cif_path, "rt") as f:
        cif = f.read()
    return cif


def matches_true(true_cif, gen_cif, struct_matcher):
    if gen_cif is None:
        return False
    true_struct = Structure.from_str(true_cif, fmt="cif")
    gen_struct = Structure.from_str(gen_cif, fmt="cif")
    try:
        is_match = struct_matcher.fit(true_struct, gen_struct)
    except Exception as e:
        print(e)
        is_match = False
    return is_match


def get_best_match(true_cif, challenge_path, formula, struct_matcher):
    results_path = os.path.join(challenge_path, formula)
    has_match = False
    best_rmsd = float("inf")
    best_cif = None
    for fname in sorted(os.listdir(results_path)):
        if fname.endswith("cif"):
            cif_path = os.path.join(results_path, fname)
            with open(cif_path, "rt") as f:
                gen_cif = f.read()

            true_struct = Structure.from_str(true_cif, fmt="cif")
            gen_struct = Structure.from_str(gen_cif, fmt="cif")
            try:
                is_match = struct_matcher.fit(true_struct, gen_struct)
                if is_match:
                    has_match = True
                    rmsd, _ = struct_matcher.get_rms_dist(true_struct, gen_struct)
                    if rmsd < best_rmsd:
                        best_rmsd = rmsd
                        best_cif = gen_cif
            except Exception as e:
                pass

    return has_match, best_cif


def get_metrics(df, n_sims):
    min_score_iter = df[df['score'] == df['score'].min()]['iteration'].values[0]
    min_score = df['score'].min()
    mean_score = df['score'].mean()
    pct_valid = (len(df) / n_sims) * 100
    min_score_file = df[df['score'] == df['score'].min()]['file'].values[0]
    return min_score, min_score_iter, mean_score, pct_valid, min_score_file


def analyze_local_environments(cif, distance_cutoff=1., angle_cutoff=0.3, max_dist_factor=1.5):
    lgf = LocalGeometryFinder()
    lgf.setup_parameters()
    allcg = AllCoordinationGeometries()
    strategy = SimplestChemenvStrategy(
        structure_environments=None,
        distance_cutoff=distance_cutoff,
        angle_cutoff=angle_cutoff,
        additional_condition=AbstractChemenvStrategy.AC.ONLY_ACB,
        continuous_symmetry_measure_cutoff=10,
        symmetry_measure_type=AbstractChemenvStrategy.DEFAULT_SYMMETRY_MEASURE_TYPE,
    )
    structure = Structure.from_str(cif, fmt="cif")
    lgf.setup_structure(structure)
    se = lgf.compute_structure_environments(maximum_distance_factor=max_dist_factor)
    strategy.set_structure_environments(se)
    analysis_string = ""
    for eqslist in se.equivalent_sites:
        site = eqslist[0]
        isite = se.structure.index(site)
        try:
            if strategy.uniquely_determines_coordination_environments:
                ces = strategy.get_site_coordination_environments(site)
            else:
                ces = strategy.get_site_coordination_environments_fractions(site)
        except NeighborsNotComputedChemenvError:
            continue
        if ces is None:
            continue
        if len(ces) == 0:
            continue
        comp = site.species
        if strategy.uniquely_determines_coordination_environments:
            ce = ces[0]
            if ce is None:
                continue
            thecg = allcg.get_geometry_from_mp_symbol(ce[0])
            analysis_string += (
                f"Environment for site #{isite} {comp.get_reduced_formula_and_factor()[0]}"
                f" ({comp}) : {thecg.name} ({ce[0]})\n"
            )
        else:
            analysis_string += (
                f"Environments for site #{isite} {comp.get_reduced_formula_and_factor()[0]} ({comp}) : \n"
            )
            for ce in ces:
                cg = allcg.get_geometry_from_mp_symbol(ce[0])
                csm = ce[1]["other_symmetry_measures"]["csm_wcs_ctwcc"]
                analysis_string += f" - {cg.name} ({cg.mp_symbol}): {ce[2]:.2%} (csm : {csm:2f})\n"
    return analysis_string


def write_file(root_dir, formula, content, fname):
    with open(os.path.join(root_dir, formula, fname), "wt") as f:
        f.write(content)


def write_cif_and_envs(root_dir, formula, cif, name, distance_cutoff, angle_cutoff, max_dist_factor):
    write_file(root_dir, formula, cif, f"{name}.cif")
    environments = analyze_local_environments(
        cif,
        distance_cutoff=distance_cutoff,
        angle_cutoff=angle_cutoff,
        max_dist_factor=max_dist_factor,
    )
    write_file(root_dir, formula, environments, f"{name}_envs.txt")


if __name__ == '__main__':
    model = "cif_model_35"  # `{model_dir}/{model}_random/` and `{model_dir}/{model}_mcts/` must exist
    model_dir = "../out"
    challenge_set_path = "../out/ChallengeSet-v1.zip"
    alignn_energies = read_alignn_energies("../out/ChallengeSet-v1.alignn_energies.csv")
    n_sims = 1000
    sg = False  # if True, `{model_dir}/{model}_random_sg/` and `{model_dir}/{model}_mcts_sg/` must exist
    out_dir = f"../out/{model}_ChallengeSet-v1_mcts{'_sg' if sg else ''}_analysis/"
    # local environment analysis
    distance_cutoff = 1.
    angle_cutoff = 0.3
    max_dist_factor = 1.5

    formulas = get_formulas(model, model_dir, sg)
    challenge_set = read_challenge_set(challenge_set_path)

    struct_matcher = StructureMatcher(
        ltol=0.2, stol=0.3, angle_tol=5, primitive_cell=True, scale=True,
        attempt_supercell=False, comparator=ElementComparator()
    )

    if os.path.exists(out_dir) and os.path.isdir(out_dir):
        print(f"path {out_dir} exists; deleting it ...")
        shutil.rmtree(out_dir)
    print(f"creating {out_dir}")
    os.makedirs(out_dir)

    # counts of when MCTS improved over random
    validity_rate_improved = 0
    min_E_improved = 0
    mean_E_improved = 0
    mcts_min_match_true_count = 0
    mcts_any_match_true_count = 0
    random_min_match_true_count = 0
    random_any_match_true_count = 0

    print("Composition       |  Algorithm   |  best E  |  best it. |  mean E  | % valid    | min matches true? | any matches true? |")
    print("------------------|--------------|----------|-----------|----------|------------|-------------------|-------------------|")

    header = ["formula", "algorithm", "true_E", "best_E", "best_it",
              "mean_E", "pct_valid", "min_matches_true", "any_matches_true"]
    rows = []

    for formula in sorted(formulas):
        os.makedirs(os.path.join(out_dir, formula))

        true_cif = challenge_set[formula]
        alignn_E = alignn_energies[formula]

        write_cif_and_envs(out_dir, formula, true_cif, "true",
                           distance_cutoff, angle_cutoff, max_dist_factor)

        random_path = os.path.join(model_dir, f"{model}_random{'_sg' if sg else ''}")
        mcts_path = os.path.join(model_dir, f"{model}_mcts{'_sg' if sg else ''}")

        df_random = pd.read_csv(os.path.join(random_path, formula, "results.csv"))
        df_mcts = pd.read_csv(os.path.join(mcts_path, formula, "results.csv"))

        rand_min_score, best_it, rand_mean_score, rand_pct_valid, min_score_file = get_metrics(df_random, n_sims)
        min_cif = read_cif(random_path, formula, min_score_file)
        min_is_match = "yes" if matches_true(true_cif, min_cif, struct_matcher) else "no"
        any_matches, best_cif = get_best_match(true_cif, random_path, formula, struct_matcher)
        any_matches = "yes" if any_matches else "no"
        print(f"{formula:18}|  random      | {rand_min_score:.5f} |  {best_it:3}      | {rand_mean_score:8.5f} | "
              f"{rand_pct_valid:10.2f} | {min_is_match:17} | {any_matches:17} |")
        rows.append([formula, "random", f"{alignn_E:.5f}", f"{rand_min_score:.5f}", best_it,
                     f"{rand_mean_score:.5f}", f"{rand_pct_valid:.2f}", min_is_match, any_matches])

        if min_cif:
            write_cif_and_envs(out_dir, formula, min_cif, "min_gen_random",
                               distance_cutoff, angle_cutoff, max_dist_factor)
        if best_cif:
            write_cif_and_envs(out_dir, formula, best_cif, "best_gen_random",
                               distance_cutoff, angle_cutoff, max_dist_factor)

        if min_is_match == "yes":
            random_min_match_true_count += 1
        if any_matches == "yes":
            random_any_match_true_count += 1

        mcts_min_score, best_it, mcts_mean_score, mcts_pct_valid, min_score_file = get_metrics(df_mcts, n_sims)
        min_cif = read_cif(mcts_path, formula, min_score_file)
        min_is_match = "yes" if matches_true(true_cif, min_cif, struct_matcher) else "no"
        any_matches, best_cif = get_best_match(true_cif, mcts_path, formula, struct_matcher)
        any_matches = "yes" if any_matches else "no"
        print(f"ALIGNN E: {alignn_E:.5f}|  mcts        | {mcts_min_score:.5f} |  {best_it:3}      | {mcts_mean_score:8.5f} | "
              f"{mcts_pct_valid:10.2f} | {min_is_match:17} | {any_matches:17} |")
        rows.append([formula, "mcts", f"{alignn_E:.5f}", f"{mcts_min_score:.5f}", best_it,
                     f"{mcts_mean_score:.5f}", f"{mcts_pct_valid:.2f}", min_is_match, any_matches])

        if min_cif:
            write_cif_and_envs(out_dir, formula, min_cif, "min_gen_mcts",
                               distance_cutoff, angle_cutoff, max_dist_factor)
        if best_cif:
            write_cif_and_envs(out_dir, formula, best_cif, "best_gen_mcts",
                               distance_cutoff, angle_cutoff, max_dist_factor)

        if min_is_match == "yes":
            mcts_min_match_true_count += 1
        if any_matches == "yes":
            mcts_any_match_true_count += 1

        if mcts_pct_valid > rand_pct_valid:
            validity_rate_improved += 1
        if mcts_min_score < rand_min_score:
            min_E_improved += 1
        if mcts_mean_score < rand_mean_score:
            mean_E_improved += 1

        print("------------------|--------------|----------|-----------|----------|------------|-------------------|-------------------|")

    tot = len(formulas)
    print(f"MCTS improves validity rate: {validity_rate_improved}/{tot}")
    print(f"MCTS improves min E:         {min_E_improved}/{tot}")
    print(f"MCTS improves mean E:        {mean_E_improved}/{tot}")
    print("                  | MCTS  | Random |")
    print("------------------|-------|--------|")
    print(f"Min matches true  | {mcts_min_match_true_count}/{tot}  | {random_min_match_true_count}/{tot}   |")
    print(f"Any matches true  | {mcts_any_match_true_count}/{tot}  | {random_any_match_true_count}/{tot}   |")

    with open(os.path.join(out_dir, "results.csv"), "wt") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)
