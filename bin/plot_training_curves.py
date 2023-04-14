import re

import matplotlib.pyplot as plt


if __name__ == '__main__':
    results_file = "../out/cif_model_19.out.txt"

    with open(results_file, "rt") as f:
        results = f.readlines()

    # Regular expression pattern to match the iter and loss values
    iter_pattern = r'iter\s+(\d+).*loss\s+([\d.]+)'

    val_pattern = r'step\s+(\d+).*train loss\s+([\d.]+).*val loss\s+([\d.]+)'

    train_iters = []
    train_losses = []

    val_iters = []
    val_losses = []

    for line in results:
        line = line.strip()
        match = re.search(iter_pattern, line)
        if match:
            train_iters.append(int(match.group(1)))
            train_losses.append(float(match.group(2)))

        match = re.search(val_pattern, line)
        if match:
            val_iters.append(int(match.group(1)))
            val_losses.append(float(match.group(3)))

    plt.plot(train_iters, train_losses, label="training", alpha=0.8)
    plt.scatter(val_iters, val_losses, label="validation", color="darkblue", s=50)
    plt.xlabel("iteration")
    plt.ylabel("loss")
    plt.title(f"training loss for {results_file.split('/')[-1].split('.')[0]}")
    plt.legend()
    plt.show()
