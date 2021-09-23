import numpy as np
import matplotlib as plt
from scipy.stats import poisson as ps  # only used for calculating pmf of poisson distribution

def generate(n_samples, c, lambds):
    Data = np.ndarray(shape=(n_samples, ))
    K = len(c)

    for j in range(0, n_samples):
        i = np.random.choice(K, p=c)
        Data[j] = np.random.poisson(lambds[i])

    return Data


def prob_x_given_z(x, z, lambds):
    return ps.pmf(x, lambds[z])


def prob_x(x, c, lambds):
    k = len(c)
    return sum([(prob_x_given_z(x, z, lambds) * c[z]) for z in range(0, k)])


def plot_dist(X, k, t, real_cs, real_lambds):
    c, lambds = PMM_EM1(X, k, T=t)

    points = list(range(0, 40))
    expected_values = list()
    real_values = list()

    for pt in points:
        expected_values.append(prob_x(pt, c, lambds))
        real_values.append(prob_x(pt, real_cs, real_lambds))

    plt.ylabel("PMF")
    plt.plot(points, expected_values, label="Estimated Distribution")
    plt.plot(points, real_values, label="True Distribution")

    plt.legend()

    plt.title("EM Algorithm : Estimated vs Real Distribution After " + str(t) + " Iterations")

    plt.show()


def PMM_EM1(X, k, T=1000):
    n = X.shape[0]
    cs = np.ones(shape=(k, )) * (1/k)
    lambds = np.ndarray(shape=(k, ))

    for i in range(0, k):
        lambds[i] = i+1

    for t in range(0, T): # O(n*k) for every iteration
        cs, lambds = EM1_update(X, k, cs, lambds)

    return cs, lambds


def EM1_update(X, k, cs, lambds):
    n = X.shape[0]
    new_cs = np.ndarray(shape=(k, ))
    new_lambds = np.ndarray(shape=(k, ))
    auxiliary_mat = np.ndarray(shape=(n, k))
    for i in range(0, n):
        x = X[i]
        for z in range(0, k):
            auxiliary_mat[i][z] = cs[z] * prob_x_given_z(x, z, lambds)

    auxiliary_mat /= np.sum(auxiliary_mat, axis=1, keepdims=True) # dividing each row by its mean
    for z in range(0, k):
        new_lambds[z] = np.dot(auxiliary_mat[:,z], X)

    columns_sums = np.sum(auxiliary_mat, axis=0)

    new_lambds /= columns_sums

    new_cs = columns_sums / n
    print(new_cs, new_lambds)

    return new_cs, new_lambds


def Example(k, real_cs, real_lambds, nsample=1000):
    Data = generate(n_samples=nsample, c=real_cs, lambds=real_lambds)
    Ts = [100, 1000]

    for t in Ts:
        plot_dist(Data, k, t, real_cs, real_lambds)


if __name__ == "__main__":
    Example(k=3, real_cs=[0.3, 0.4, 0.3], real_lambds=[5, 10, 11])
