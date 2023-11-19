import csv
import matplotlib.pyplot as plt
from matplotlib.colors import hsv_to_rgb
from math import fmod, sqrt


def read_iteration_to_gen_counts(fname):
    with open(fname, "rt") as f:
        reader = csv.reader(f)
        next(reader)  # skip header
        iterations_to_gen_counts = {}
        n = 0
        for line in reader:
            n += 1
            iteration = int(line[1])
            iterations_to_gen_counts[iteration] = n
    return iterations_to_gen_counts


if __name__ == '__main__':
    model = "cif_model_35"
    mcts_name = "mcts_by"
    random_name = "random"
    formulas = [
        "Ba2Fe2F9",
        "Ba2Gd(BO3)2F",
        "Ba4GeSb2Se11",
        "Ba6Fe2Te3S7",
        "Ba9Yb2(SiO4)6",
        "CaHPO3",
        "CH3NH3PbI3",
        "Cs2Al2O3F2",
        "HgB2S4",
        "La4Ga2S8O3",
        "Li9Al4Sn5",
        "LiBa2AlO4",
        "Mn4(PO4)3",
        "Na4Sn2Ge5O16",
        "Na5Mn4P4H4(O9F2)2",
        "NaSb2TeO7",
        "NaSbSe2O7",
        "RbNiFe(PO4)2",
        "Sr6Ge3OSe11",
        "SrCo4(OH)(PO4)3",
    ]
    n_steps = 1000
    out_fname = "../out/valid_struct_count.pdf"

    plt.rcParams["figure.figsize"] = (15, 7)

    colors = []
    # from https://gamedev.stackexchange.com/questions/46463/how-can-i-find-an-optimum-set-of-colors-for-10-players/46469#46469
    for f in range(len(formulas)):
        colors.append(hsv_to_rgb((fmod(f * 0.618033988749895, 1.0), 1., sqrt(1.0 - fmod(f * 0.618033988749895, 0.5)))))

    for c, formula in enumerate(formulas):
        mcts_results_csv_fname = f"../out/{model}_{mcts_name}/{formula}/results.csv"
        random_results_csv_fname = f"../out/{model}_{random_name}/{formula}/results.csv"
        color = colors[c]

        mcts_iterations_to_gen_counts = read_iteration_to_gen_counts(mcts_results_csv_fname)
        random_iterations_to_gen_counts = read_iteration_to_gen_counts(random_results_csv_fname)

        mcts_curr_gen_count = 0
        random_curr_gen_count = 0
        X = []
        Y_mcts = []
        Y_random = []
        for i in range(1, n_steps+1):
            X.append(i)
            if i in mcts_iterations_to_gen_counts:
                mcts_curr_gen_count = mcts_iterations_to_gen_counts[i]
            Y_mcts.append(mcts_curr_gen_count)

            if i in random_iterations_to_gen_counts:
                random_curr_gen_count = random_iterations_to_gen_counts[i]
            Y_random.append(random_curr_gen_count)

        plt.plot(X, Y_mcts, label=f"{formula}", alpha=0.6, color=color)
        plt.plot(X, Y_random, alpha=0.6, linestyle="dashed", color=color)

    xticks = list(range(100, 1001, 100))
    xticks.insert(0, 1)
    plt.xticks(xticks, xticks)
    plt.xlabel("iteration")
    plt.ylabel("# valid generations")

    legend1 = plt.legend(loc='upper left')
    custom_lines = [
        plt.Line2D([0], [0], color="black", linestyle="solid", alpha=0.6),
        plt.Line2D([0], [0], color="black", linestyle="dashed", alpha=0.6),
    ]
    plt.legend(custom_lines, ["MCTS", "Random"], loc="upper center")
    plt.gca().add_artist(legend1)

    plt.savefig(out_fname)
    plt.show()
