import re

import matplotlib.pyplot as plt

LOSS_PATTERN = r'step (\d+): train loss ([\d.]+), val loss ([\d.]+)'


def get_data(results):
    train_iters = []
    train_losses = []
    val_losses = []
    for line in results:
        line = line.strip()
        match = re.search(LOSS_PATTERN, line)
        if match:
            train_iters.append(int(match.group(1)))
            train_losses.append(float(match.group(2)))
            val_losses.append(float(match.group(3)))
    return train_iters, train_losses, val_losses


def get_lowest_val_iters(iters, losses):
    assert len(iters) == len(losses), "iters and losses must have the same length"
    lowest_iters = []
    lowest_losses = []
    for i in range(len(iters)):
        it = iters[i]
        loss = losses[i]
        if len(lowest_losses) == 0 or loss < lowest_losses[-1]:
            lowest_iters.append(it)
            lowest_losses.append(loss)
    return lowest_iters, lowest_losses


if __name__ == '__main__':
    results_file, ylim = "../out/cif_model_35val.out.txt", (0.1791, 0.2589)
    out_fname = "../out/train_vs_val_error.pdf"

    with open(results_file, "rt") as f:
        results = f.readlines()

    train_iters, train_losses, val_losses = get_data(results)

    lowest_iters, lowest_losses = get_lowest_val_iters(train_iters, val_losses)
    for i, it in enumerate(train_iters):
        s = f"iter {it}: train loss: {train_losses[i]}, val loss: {val_losses[i]}"
        if it in lowest_iters:
            s += " <- best"
        print(s)

    plt.rcParams["figure.figsize"] = (10, 6)

    plt.plot(
        train_iters,
        train_losses,
        label="train",
        alpha=0.4
    )
    plt.plot(
        train_iters,
        val_losses,
        label="validation",
        alpha=0.4
    )
    plt.scatter(
        lowest_iters,
        lowest_losses,
        label="best iter",
        color="green",
        alpha=0.4,
    )

    plt.xlabel("iteration")
    plt.ylabel("loss")
    plt.ylim(ylim)
    plt.legend()

    plt.savefig(out_fname)
    plt.show()
