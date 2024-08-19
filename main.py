import numpy as np
from scipy.optimize import minimize
from typing import Tuple


def create_basic_allais() -> Tuple[np.array, np.array, np.array]:
    M = np.array([[13.911, 13.911, 13.911], [11.513, 13.911, 15.445]])
    p = np.array([0.01, 0.89, 0.1])
    z = np.array([13.911, 13.5])  # slight preference for the sure thing
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
    z = np.array([13.911, 13.5, 11.5, 11.7])
    return M, p, z


def _loss(v, p):
    return sum(np.square(v - p))


def jac(v):
    return 2 * v


def get_q(M, p, z):

    x0 = np.random.uniform(0.0, 1.0, len(p))
    cons = [
        {
            "type": "ineq",
            "fun": lambda x: x - np.ones(len(x)) * min(p),
            "jac": lambda x: np.diag(np.ones(len(x))),
        },
        {"type": "eq", "fun": lambda x: np.matmul(M, x) - z, "jac": lambda x: M},
    ]
    result = minimize(
        lambda x: _loss(x, p),
        x0,
        jac=jac,
        constraints=cons,
        method="SLSQP",
        options={
            "disp": False,
            "ftol": 1,
        },  # huge tolerance, don't need to be that close
    )
    q = result.x / sum(result.x)
    utility = np.matmul(M, q)
    if result.success:
        return q, utility
    else:
        print(result)
        print("q", q)
        print("utility", utility)
        raise Exception("Failed to converge")


if __name__ == "__main__":
    M, p, z = create_basic_allais()
    q_basic, result_basic = get_q(M, p, z)
    print(q_basic)
    print(result_basic)

    M, p, z = create_complex_allais()
    q_complex, result_complex = get_q(M, p, z)
    print(q_complex)
    print(result_complex)
