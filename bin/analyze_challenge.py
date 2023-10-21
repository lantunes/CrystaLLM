import os.path
import shutil
import csv
from zipfile import ZipFile
from io import StringIO
from pymatgen.core import Structure
from pymatgen.analysis.structure_matcher import StructureMatcher, ElementComparator
from pymatgen.analysis.chemenv.coordination_environments.chemenv_strategies import SimplestChemenvStrategy, \
    AbstractChemenvStrategy
from pymatgen.analysis.chemenv.coordination_environments.coordination_geometries import AllCoordinationGeometries
from pymatgen.analysis.chemenv.coordination_environments.coordination_geometry_finder import LocalGeometryFinder
from pymatgen.analysis.chemenv.utils.chemenv_errors import NeighborsNotComputedChemenvError

import warnings
warnings.filterwarnings("ignore")


def read_challenge_set(challenge_set_path):
    input_zip = ZipFile(challenge_set_path)
    challenge_set = {}
    training_set_formulas = set()

    for zipfile in input_zip.filelist:
        components = zipfile.filename.split("/")

        if components[-1] == "metadata.csv":
            content = input_zip.read(zipfile.filename).decode("utf-8")
            f = StringIO(content)
            reader = csv.reader(f)
            next(reader)  # skip header
            for line in reader:
                formula = line[0].strip()
                source = line[1].strip()
                if source == "training set":
                    training_set_formulas.add(formula)
            continue

        if len(components) < 3 or len(components[-1]) == 0:
            continue

        formula = components[1]
        fname = components[2]
        if fname.endswith("pymatgen.cif"):
            content = input_zip.read(zipfile.filename).decode("utf-8")
            challenge_set[formula] = content

    return challenge_set, training_set_formulas


def get_min_cif(challenge_path, formula):
    results_path = os.path.join(challenge_path, formula, "results.csv")
    min_score = float("inf")
    min_cif_fname = None
    with open(results_path, "rt") as f:
        reader = csv.reader(f)
        next(reader)  # skip header
        for line in reader:
            cif_fname = line[0]
            score = float(line[2])
            if score < min_score:
                min_score = score
                min_cif_fname = cif_fname
    min_cif_path = os.path.join(challenge_path, formula, min_cif_fname)
    with open(min_cif_path, "rt") as f:
        min_cif = f.read()
    return min_cif


def read_results_csv(challenge_path):
    results_path = os.path.join(challenge_path, "results.csv")
    # map from formula -> {"validity_rate": <float>, "mean_E": <float>, "min_E": <float>, "min_cif": <str>}
    results = {}
    with open(results_path, "rt") as f:
        reader = csv.reader(f)
        next(reader)  # skip header
        for line in reader:
            formula = line[0]
            validity_rate = float(line[1])
            mean_E = float(line[2])
            min_E = float(line[3])
            min_cif = get_min_cif(challenge_path, formula) if validity_rate > 0 else None
            results[formula] = {
                "validity_rate": validity_rate,
                "mean_E": mean_E,
                "min_E": min_E,
                "min_cif": min_cif,
            }
    return results


def read_props(fname, formula):
    results = read_results_csv(fname)
    validity_rate = results[formula]["validity_rate"] if formula in results else 0.0
    min_E = results[formula]["min_E"] if formula in results else float("nan")
    mean_E = results[formula]["mean_E"] if formula in results else float("nan")
    min_cif = results[formula]["min_cif"] if formula in results else None
    return validity_rate, min_E, mean_E, min_cif


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


def is_valid_on_first(formula_dir):
    fname = os.path.join(formula_dir, "results.csv")
    if not os.path.exists(fname):
        return False
    with open(fname, "rt") as f:
        reader = csv.reader(f)
        next(reader)  # skip header
        first_line = next(reader)
        iteration = int(first_line[1])
        if iteration == 1:
            return True
    return False


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


if __name__ == '__main__':
    model = "cif_model_35"
    model_dir = "../out"
    challenge_set_path = "../out/ChallengeSet-v1.zip"
    alignn_energies = read_alignn_energies("../out/ChallengeSet-v1.alignn_energies.csv")
    out_dir = "../out/cif_model_35_ChallengeSet-v1_analysis/"
    # local environment analysis
    distance_cutoff = 1.
    angle_cutoff = 0.3
    max_dist_factor = 1.5

    challenge_set, training_set_formulas = read_challenge_set(challenge_set_path)

    struct_matcher = StructureMatcher(
        ltol=0.2, stol=0.3, angle_tol=5, primitive_cell=True, scale=True,
        attempt_supercell=False, comparator=ElementComparator()
    )

    if os.path.exists(out_dir) and os.path.isdir(out_dir):
        print(f"path {out_dir} exists; deleting it ...")
        shutil.rmtree(out_dir)
    print(f"creating {out_dir}")
    os.makedirs(out_dir)

    validity_count = [0, 0]  # [no space group, w/ space group]
    valid_on_first_count = [0, 0]
    min_match_true_count_all = [0, 0]
    min_match_true_count_unseen = [0, 0]
    any_match_true_count_all = [0, 0]
    any_match_true_count_unseen = [0, 0]
    n_unseen = 0

    print("Composition         | space group? |  mean E  |  min E   | % valid | valid on first? | min matches true? | any matches true? |")
    print("--------------------|--------------|----------|----------|---------|-----------------|-------------------|-------------------|")

    header = ["formula", "seen_in_training", "true_E", "includes_space_group",
              "mean_E", "min_E", "pct_valid", "valid_on_first", "min_matches_true", "any_matches_true"]
    rows = []

    for formula in sorted(challenge_set):

        os.makedirs(os.path.join(out_dir, formula))

        true_cif = challenge_set[formula]
        alignn_E = alignn_energies[formula]
        if formula not in training_set_formulas:
            n_unseen += 1
            seen = ""
        else:
            seen = "* "

        write_cif_and_envs(out_dir, formula, true_cif, "true",
                           distance_cutoff, angle_cutoff, max_dist_factor)

        challenge_path = os.path.join(model_dir, f"{model}_challenge")
        validity_rate, min_E, mean_E, min_cif = read_props(challenge_path, formula)
        valid_on_first = "yes" if is_valid_on_first(os.path.join(challenge_path, formula)) else "no"
        min_is_match = "yes" if matches_true(true_cif, min_cif, struct_matcher) else "no"
        any_matches, best_cif = get_best_match(true_cif, challenge_path, formula, struct_matcher)
        any_matches = "yes" if any_matches else "no"
        print(f"{seen + formula:20}|      no      | {mean_E:8.5f} | {min_E:8.5f} | "
              f"{validity_rate:7.2f} | {valid_on_first:15} | {min_is_match:17} | {any_matches:17} |")

        if min_cif:
            write_cif_and_envs(out_dir, formula, min_cif, "min_gen_no_spacegroup",
                               distance_cutoff, angle_cutoff, max_dist_factor)

        if best_cif:
            write_cif_and_envs(out_dir, formula, best_cif, "best_gen_no_spacegroup",
                               distance_cutoff, angle_cutoff, max_dist_factor)

        rows.append([
            formula,
            "yes" if formula in training_set_formulas else "no",
            f"{alignn_E:.5f}",
            "no",
            f"{mean_E:.5f}",
            f"{min_E:.5f}",
            f"{validity_rate:.2f}",
            valid_on_first,
            min_is_match,
            any_matches,
        ])

        if validity_rate > 0:
            validity_count[0] += 1
        if valid_on_first == "yes":
            valid_on_first_count[0] += 1
        if min_is_match == "yes":
            min_match_true_count_all[0] += 1
            if not seen.startswith("*"):
                min_match_true_count_unseen[0] += 1
        if any_matches == "yes":
            any_match_true_count_all[0] += 1
            if not seen.startswith("*"):
                any_match_true_count_unseen[0] += 1

        challenge_path = os.path.join(model_dir, f"{model}_challenge_sg")
        validity_rate, min_E, mean_E, min_cif = read_props(challenge_path, formula)
        valid_on_first = "yes" if is_valid_on_first(os.path.join(challenge_path, formula)) else "no"
        min_is_match = "yes" if matches_true(true_cif, min_cif, struct_matcher) else "no"
        any_matches, best_cif = get_best_match(true_cif, challenge_path, formula, struct_matcher)
        any_matches = "yes" if any_matches else "no"
        print(f"ALIGNN E: {alignn_E:8.5f}  |      yes     | {mean_E:8.5f} | {min_E:8.5f} | "
              f"{validity_rate:7.2f} | {valid_on_first:15} | {min_is_match:17} | {any_matches:17} |")

        if min_cif:
            write_cif_and_envs(out_dir, formula, min_cif, "min_gen_with_spacegroup",
                               distance_cutoff, angle_cutoff, max_dist_factor)

        if best_cif:
            write_cif_and_envs(out_dir, formula, best_cif, "best_gen_with_spacegroup",
                               distance_cutoff, angle_cutoff, max_dist_factor)

        rows.append([
            formula,
            "yes" if formula in training_set_formulas else "no",
            f"{alignn_E:.5f}",
            "yes",
            f"{mean_E:.5f}",
            f"{min_E:.5f}",
            f"{validity_rate:.2f}",
            valid_on_first,
            min_is_match,
            any_matches,
        ])

        if validity_rate > 0:
            validity_count[1] += 1
        if valid_on_first == "yes":
            valid_on_first_count[1] += 1
        if min_is_match == "yes":
            min_match_true_count_all[1] += 1
            if not seen.startswith("*"):
                min_match_true_count_unseen[1] += 1
        if any_matches == "yes":
            any_match_true_count_all[1] += 1
            if not seen.startswith("*"):
                any_match_true_count_unseen[1] += 1

        print("--------------------|--------------|----------|----------|---------|-----------------|-------------------|-------------------|")

    print("* seen in training")

    tot = len(challenge_set)
    print( "                          | no sg | w/ sg |")
    print( "--------------------------|-------|-------|")
    print(f"Can generate              | {validity_count[0]}/{tot} | {validity_count[1]}/{tot} |")
    print(f"Valid on first            | {valid_on_first_count[0]}/{tot} | {valid_on_first_count[1]}/{tot} |")
    print(f"Min matches true (all)    | {min_match_true_count_all[0]}/{tot} | {min_match_true_count_all[1]}/{tot} |")
    print(f"Min matches true (unseen) | {min_match_true_count_unseen[0]}/{n_unseen} | {min_match_true_count_unseen[1]}/{n_unseen} |")
    print(f"Any matches true (all)    | {any_match_true_count_all[0]}/{tot} | {any_match_true_count_all[1]}/{tot} |")
    print(f"Any matches true (unseen) | {any_match_true_count_unseen[0]}/{n_unseen} | {any_match_true_count_unseen[1]}/{n_unseen} |")

    with open(os.path.join(out_dir, "results.csv"), "wt") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)
