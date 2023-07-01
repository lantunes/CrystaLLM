import csv
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score


if __name__ == '__main__':
    results_fname = "../out/cif_model_24.evalresults-sg.csv"
    matches_fname = "../out/cif_model_24.evalresults-sg-match.csv"
    generations = 3
    write_first_match_results = False
    first_match_results_lengths_out_fname = "../out/cif_model_24.evalresults-sg.first_match_lengths.csv"
    first_match_results_volume_out_fname = "../out/cif_model_24.evalresults-sg.first_match_volume.csv"

    formula_sg_to_matches = {}
    with open(matches_fname, "rt") as f:
        reader = csv.DictReader(f, delimiter=",")
        for row in reader:
            formula = row["formula"]
            sg = row["sg"]
            matches = []
            for i in range(1, generations+1):
                matches.append(bool(int(row[f"gen{i}_match"])))
            formula_sg_to_matches[f"{formula}-{sg}"] = matches

    all_true_cell_lengths = []
    all_generated_cell_lengths = {i: [] for i in range(1, generations+1)}
    all_true_volumes = []
    all_generated_volumes = {i: [] for i in range(1, generations+1)}
    all_implied_volumes = {i: [] for i in range(1, generations+1)}

    first_match_true_cell_lengths = []
    first_match_generated_cell_lengths = []
    first_match_true_volumes = []
    first_match_generated_volumes = []
    first_match_implied_volumes = []

    with open(results_fname, "rt") as f:
        reader = csv.DictReader(f, delimiter=",")
        for row in reader:
            formula = row["formula"]
            sg = row["sg"]

            all_true_cell_lengths.append(float(row["true_a"]))
            all_true_cell_lengths.append(float(row["true_b"]))
            all_true_cell_lengths.append(float(row["true_c"]))

            all_true_volumes.append(float(row["true_vol"]))

            for i in range(1, generations+1):
                all_generated_cell_lengths[i].append(float(row[f"gen{i}_a"]))
                all_generated_cell_lengths[i].append(float(row[f"gen{i}_b"]))
                all_generated_cell_lengths[i].append(float(row[f"gen{i}_c"]))

                all_generated_volumes[i].append(float(row[f"gen{i}_vol_generated"]))

                all_implied_volumes[i].append(float(row[f"gen{i}_vol_implied"]))

            matched = formula_sg_to_matches[f"{formula}-{sg}"]
            if True in matched:
                match_idx = matched.index(True)

                first_match_generated_cell_lengths.append(float(row[f"gen{match_idx+1}_a"]))
                first_match_generated_cell_lengths.append(float(row[f"gen{match_idx+1}_b"]))
                first_match_generated_cell_lengths.append(float(row[f"gen{match_idx+1}_c"]))

                first_match_true_cell_lengths.append(float(row["true_a"]))
                first_match_true_cell_lengths.append(float(row["true_b"]))
                first_match_true_cell_lengths.append(float(row["true_c"]))

                first_match_generated_volumes.append(float(row[f"gen{match_idx+1}_vol_generated"]))

                first_match_true_volumes.append(float(row["true_vol"]))

                first_match_implied_volumes.append(float(row[f"gen{match_idx+1}_vol_implied"]))

    print(f"All generated (across {generations} generation attempts):")

    r2_cell_lengths_vs_true = [
        r2_score(all_true_cell_lengths, all_generated_cell_lengths[i]) for i in range(1, generations+1)
    ]
    mae_cell_lengths_vs_true = [
        mean_absolute_error(all_true_cell_lengths, all_generated_cell_lengths[i]) for i in range(1, generations + 1)
    ]
    print(f"cell lengths (vs. true): R2: {np.mean(r2_cell_lengths_vs_true):.3f} ± {np.std(r2_cell_lengths_vs_true):.3f}, "
          f"MAE: {np.mean(mae_cell_lengths_vs_true):.3f} ± {np.std(mae_cell_lengths_vs_true):.3f}")

    r2_vol_vs_true = [
        r2_score(all_true_volumes, all_generated_volumes[i]) for i in range(1, generations + 1)
    ]
    mae_vol_vs_true = [
        mean_absolute_error(all_true_volumes, all_generated_volumes[i]) for i in range(1, generations + 1)
    ]
    print(f"volume (vs. true): R2: {np.mean(r2_vol_vs_true):.3f} ± {np.std(r2_vol_vs_true):.3f}, "
          f"MAE: {np.mean(mae_vol_vs_true):.3f} ± {np.std(mae_vol_vs_true):.3f}")

    r2_vol_vs_implied = [
        r2_score(all_implied_volumes[i], all_generated_volumes[i]) for i in range(1, generations + 1)
    ]
    mae_vol_vs_implied = [
        mean_absolute_error(all_implied_volumes[i], all_generated_volumes[i]) for i in range(1, generations + 1)
    ]
    print(f"volume (vs. implied): R2: {np.mean(r2_vol_vs_implied):.3f} ± {np.std(r2_vol_vs_implied):.3f}, "
          f"MAE: {np.mean(mae_vol_vs_implied):.3f} ± {np.std(mae_vol_vs_implied):.3f}")

    non_matching_pct = 100-((len(first_match_true_cell_lengths)/3) / len(formula_sg_to_matches))*100
    print(f"First match for each entry with a match only ({non_matching_pct:.0f}% have no matches):")

    r2_cell_lengths_vs_true = r2_score(first_match_true_cell_lengths, first_match_generated_cell_lengths)
    mae_cell_lengths_vs_true = mean_absolute_error(first_match_true_cell_lengths, first_match_generated_cell_lengths)
    print(f"cell lengths (vs. true): R2: {r2_cell_lengths_vs_true:.3f}, MAE: {mae_cell_lengths_vs_true:.3f}")

    r2_vol_vs_true = r2_score(first_match_true_volumes, first_match_generated_volumes)
    mae_vol_vs_true = mean_absolute_error(first_match_true_volumes, first_match_generated_volumes)
    print(f"volume (vs. true): R2: {r2_vol_vs_true:.3f}, MAE: {mae_vol_vs_true:.3f}")

    r2_vol_vs_true = r2_score(first_match_implied_volumes, first_match_generated_volumes)
    mae_vol_vs_true = mean_absolute_error(first_match_implied_volumes, first_match_generated_volumes)
    print(f"volume (vs. implied): R2: {r2_vol_vs_true:.3f}, MAE: {mae_vol_vs_true:.3f}")

    if write_first_match_results:
        df_lengths = pd.DataFrame({
            "true_lengths": first_match_true_cell_lengths,
            "gen_lengths": first_match_generated_cell_lengths,
        })
        df_lengths.to_csv(first_match_results_lengths_out_fname)

        df_volume = pd.DataFrame({
            "true_volume": first_match_true_volumes,
            "gen_volume": first_match_generated_volumes,
            "implied_volume": first_match_implied_volumes,
        })
        df_volume.to_csv(first_match_results_volume_out_fname)
