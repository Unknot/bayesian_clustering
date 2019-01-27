import numpy as np
from pprint import pprint
import matplotlib.pyplot as plt

classes = {
    "A": {"n": 20, "mean": np.array([-10., 0.]), "cov": np.array([(1**2, 0.), (0., 1**2)], dtype=np.float)},
    "B": {"n": 20, "mean": np.array([0., 5.]), "cov": np.array([(1.**2, 0.), (0., 1.**2)], dtype=np.float)},
    "C": {"n": 20, "mean": np.array([10., -5.]), "cov": np.array([(1**2, 0.), (0., 1**2)], dtype=np.float)},
    "D": {"n": 15, "mean": np.array([2., -2.]), "cov": np.array([(1**2, 0.), (0., 1**2)], dtype=np.float)},
    "E": {"n": 50, "mean": np.array([6., -15.]), "cov": np.array([(1**2, 0.), (0., 1**2)], dtype=np.float)}
}


def main():
    sample = np.empty((0, 2))
    for (k, v) in classes.items():
        pprint(k)
        sample = np.vstack((sample, np.random.multivariate_normal(v["mean"], v["cov"], v["n"])))

    plt.plot([i[0] for i in sample], [i[1] for i in sample], 'o')
    plt.show()
    X = np.matrix(sample).T
    
    var_covar = (X.T @ X)
    pprint(var_covar.shape)
    
    np.savetxt("varcovar.csv", var_covar, delimiter=",")
    
    plt.imshow(var_covar, cmap='hot', interpolation='nearest')
    plt.show()


if __name__ == "__main__":
    main()
