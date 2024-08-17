import numpy as np
from scipy.optimize import nnls
from typing import Tuple


def create_basic_allais() -> Tuple[np.array, np.array, np.array]:
    M = np.array([[13.911, 13.911, 13.911], [11.513, 13.911, 15.445]])
    p = np.array([0.01, 0.89, 0.1])
    z = np.array([2, 1])  # slight preference for the sure thing
    return M, p, z


def create_complex_allais() -> Tuple[np.array, np.array, np.array]:
    M = np.array(
        [
            [
                13.911,
                13.911,
                13.911,
                13.911,
                13.911,
                13.911,
                13.911,
                13.911,
                13.911,
                13.911,
                13.911,
                13.911,
            ],
            [
                11.513,
                13.911,
                15.445,
                11.513,
                13.911,
                15.445,
                11.513,
                13.911,
                15.445,
                11.513,
                13.911,
                15.445,
            ],
            [
                11.513,
                11.513,
                11.513,
                13.911,
                13.911,
                13.911,
                11.513,
                11.513,
                11.513,
                13.911,
                13.911,
                13.911,
            ],
            [
                11.513,
                11.513,
                11.513,
                11.513,
                11.513,
                11.513,
                15.445,
                15.445,
                15.445,
                15.445,
                15.445,
                15.445,
            ],
        ]
    )
    p = np.array(
        [
            0.00801,
            0.71289,
            0.08010,
            0.00099,
            0.08811,
            0.00990,
            0.00089,
            0.07921,
            0.00890,
            0.00011,
            0.00979,
            0.00110,
        ]
    )
    z = np.array([5, 2, 2, 3])
    return M, p, z


def get_q(M, p, z, lmbda):
    n = len(p)
    m = len(z)
    MP = np.matmul(M, np.diag(p))
    lmbda_mat = np.diag(np.full(n, lmbda))
    unif = np.ones(n) * (lmbda / n)
    padded_MP = np.pad(MP, ((0, n - m), (0, 0)))
    padded_z = np.pad(z, (0, n - m))
    x_approx_sol, x_residuals = nnls(padded_MP + lmbda_mat, padded_z + unif)
    q = x_approx_sol * p
    q = q / sum(q)
    return (x_approx_sol, q, np.matmul(M, q))


if __name__ == "__main__":
    M, p, z = create_basic_allais()
    lmbda = 0.3
    sol, q_basic, result_basic = get_q(M, p, z, lmbda)
    print(sol)
    print(q_basic)
    print(result_basic)

    M, p, z = create_complex_allais()
    lmbda = 0.5
    sol, q_complex, result_complex = get_q(M, p, z, lmbda)
    print(sol)
    print(q_complex)
    print(result_complex)
