import sys
sys.path.append(".")
import argparse
import csv

from lib import CIFData, populate_cif_data
from subprocess import Popen, PIPE
from sklearn.metrics import mean_absolute_error, r2_score
from tqdm import tqdm


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Read in arguments for your script")

    parser.add_argument('--model_dir', type=str, required=True, help='Model directory')
    parser.add_argument('--eval_file', type=str, required=True, help='eval.csv file')
    parser.add_argument('--symmetrized', action='store_true', default=False, help='Symmetrized flag')
    parser.add_argument('--top_k', type=int, default=10, help='Top K value')
    parser.add_argument('--max_new_tokens', type=int, default=3000, help='Maximum new tokens')
    parser.add_argument('--device', type=str, default='cuda', choices=['cuda', 'cpu'], help='Device to use')

    args = parser.parse_args()

    model_dir = args.model_dir
    eval_fname = args.eval_file
    symmetrized = args.symmetrized
    top_k = args.top_k
    max_new_tokens = args.max_new_tokens
    device = args.device

    n_evaluations = 0
    n_failures = 0

    cell_length_a = {"true": [], "predicted": []}
    cell_length_b = {"true": [], "predicted": []}
    cell_length_c = {"true": [], "predicted": []}
    cell_angle_alpha = {"true": [], "predicted": []}
    cell_angle_beta = {"true": [], "predicted": []}
    cell_angle_gamma = {"true": [], "predicted": []}
    cell_volume = {"true": [], "predicted": []}

    with open(eval_fname, "rt") as f:
        reader = csv.reader(f)
        next(reader)  # skip header
        for line in tqdm(reader):
            cif_data = CIFData.from_csv_row(line)

            start = f"data_{cif_data.composition}\n"

            sample_args = [
                "python", "nanoGPT/sample.py", f"--device={device}", f"--out_dir={model_dir}",
                f"--start={start}", "--num_samples=1", f"--top_k={top_k}",
                f"--max_new_tokens={max_new_tokens}", f"--symmetrized={symmetrized}"
            ]
            process = Popen(sample_args, stdout=PIPE)
            output, err = process.communicate()
            exit_code = process.wait()
            if exit_code != 0:
                print(f"non-zero exit code: {exit_code}")
                print("".join(sample_args))
                print(err)
                sys.exit(1)

            n_evaluations += 1

            predicted_data = CIFData()
            try:
                # parse the output
                populate_cif_data(predicted_data, output.decode("utf-8"))
            except Exception:
                n_failures += 1
                continue

            cell_length_a["true"].append(cif_data.cell_length_a)
            cell_length_a["predicted"].append(predicted_data.cell_length_a)

            cell_length_b["true"].append(cif_data.cell_length_b)
            cell_length_b["predicted"].append(predicted_data.cell_length_b)

            cell_length_c["true"].append(cif_data.cell_length_c)
            cell_length_c["predicted"].append(predicted_data.cell_length_c)

            cell_angle_alpha["true"].append(cif_data.cell_angle_alpha)
            cell_angle_alpha["predicted"].append(predicted_data.cell_angle_alpha)

            cell_angle_beta["true"].append(cif_data.cell_angle_beta)
            cell_angle_beta["predicted"].append(predicted_data.cell_angle_beta)

            cell_angle_gamma["true"].append(cif_data.cell_angle_gamma)
            cell_angle_gamma["predicted"].append(predicted_data.cell_angle_gamma)

            cell_volume["true"].append(cif_data.cell_volume)
            cell_volume["predicted"].append(predicted_data.cell_volume)

    print(f"evalutations: {n_evaluations:,}, failures: {n_failures:,}")

    print(f"cell_length_a: "
          f"MAE: {mean_absolute_error(cell_length_a['true'], cell_length_a['predicted']):.4f}, "
          f"R2: {r2_score(cell_length_a['true'], cell_length_a['predicted']):.2f}")

    print(f"cell_length_b: "
          f"MAE: {mean_absolute_error(cell_length_b['true'], cell_length_b['predicted']):.4f}, "
          f"R2: {r2_score(cell_length_b['true'], cell_length_b['predicted']):.2f}")

    print(f"cell_length_c: "
          f"MAE: {mean_absolute_error(cell_length_c['true'], cell_length_c['predicted']):.4f}, "
          f"R2: {r2_score(cell_length_c['true'], cell_length_c['predicted']):.2f}")

    print(f"cell_angle_alpha: "
          f"MAE: {mean_absolute_error(cell_angle_alpha['true'], cell_angle_alpha['predicted']):.4f}, "
          f"R2: {r2_score(cell_angle_alpha['true'], cell_angle_alpha['predicted']):.2f}")

    print(f"cell_angle_beta: "
          f"MAE: {mean_absolute_error(cell_angle_beta['true'], cell_angle_beta['predicted']):.4f}, "
          f"R2: {r2_score(cell_angle_beta['true'], cell_angle_beta['predicted']):.2f}")

    print(f"cell_angle_gamma: "
          f"MAE: {mean_absolute_error(cell_angle_gamma['true'], cell_angle_gamma['predicted']):.4f}, "
          f"R2: {r2_score(cell_angle_gamma['true'], cell_angle_gamma['predicted']):.2f}")

    print(f"cell_volume: "
          f"MAE: {mean_absolute_error(cell_volume['true'], cell_volume['predicted']):.4f}, "
          f"R2: {r2_score(cell_volume['true'], cell_volume['predicted']):.2f}")
