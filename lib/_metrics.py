import numpy as np
import warnings


def abs_r_score(actual, predicted):
    """
    An example comparison between |R| and R^2:
    ```
    import matplotlib.pyplot as plt
    from sklearn.metrics import r2_score

    actual =    np.array([1, 4, 1, 3, 6, 4, 5, 1, 2, 5])
    predicted = np.array([1, 3, 1, 2, 1, 4, 4, 1, 3, 5])

    print(abs_r_score(actual, predicted))
    print(r2_score(actual, predicted))

    plt.scatter(predicted, actual)
    plt.yticks(list(range(7)))
    plt.xticks(list(range(7)))
    plt.show()
    ```
    """
    actual = np.array(actual)
    predicted = np.array(predicted)

    if len(predicted) < 2:
        msg = "|R| score is not well-defined with less than two samples."
        warnings.warn(msg, UserWarning)
        return float("nan")

    # sum of the absolute errors
    sae = np.sum(np.abs(actual - predicted))

    # sum of the absolute deviations from the mean
    sad = np.sum(np.abs(actual - np.mean(actual)))

    return 1 - (sae / sad)
