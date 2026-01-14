from typing import Callable
import itertools
import math


def square(x: float) -> float:
    return x * x


def get_best_quadratic_probabilities(
    gamble_outcomes: list[float],
    gamble_probabilities: list[float],
    equivalent_utility: float,
    utility: Callable[[float], float],
) -> list[float]:
    n = len(gamble_outcomes)
    if n == 1:
        return [1.0]
    real_world_expected_value = sum(
        utility(outcome) * probability
        for outcome, probability in zip(gamble_outcomes, gamble_probabilities)
    )
    total_gamble_utility = sum(utility(outcome) for outcome in gamble_outcomes)
    total_gamble_utility_sq = sum(
        square(utility(outcome)) for outcome in gamble_outcomes
    )

    lambda1 = (
        2
        * (equivalent_utility - real_world_expected_value)
        / (square(total_gamble_utility) / n - total_gamble_utility_sq)
    )
    lambda2 = -lambda1 * total_gamble_utility / n
    return [
        probability - utility(outcome) * lambda1 * 0.5 - lambda2 * 0.5
        for outcome, probability in zip(gamble_outcomes, gamble_probabilities)
    ]


def utility(init_wealth: float, result: float) -> float:
    return math.log(init_wealth + result)


def get_probabilities(probabilities: list[list[float]]) -> list[float]:
    return [math.prod(v) for v in itertools.product(*probabilities)]


if __name__ == "__main__":
    gambles: list[list[float]] = [
        [1000000.0],
        [0.0, 1000000.0, 5000000.0],
        [0.0, 1000000.0],
        [0.0, 5000000.0],
    ]
    probabilities = [[1], [0.01, 0.89, 0.1], [0.89, 0.11], [0.9, 0.1]]
    init_wealth = 100000
    equivalent_utilities = [
        utility(init_wealth, 1000000),
        utility(init_wealth, 988000),
        utility(init_wealth, 5000),
        utility(init_wealth, 5500),
    ]

    ## real work probability
    p = get_probabilities(probabilities)
    print("p")
    print(p)

    rn_prob = [
        get_best_quadratic_probabilities(
            gamble, probs, eq_utility, lambda x: utility(init_wealth, x)
        )
        for gamble, probs, eq_utility in zip(
            gambles, probabilities, equivalent_utilities
        )
    ]
    for index, (prob, gamble) in enumerate(zip(rn_prob, gambles)):
        print(f"sum of q_{index + 1} (should be one)", sum(prob))  # should all equal 1
        print(
            f"expected utility of gamble {index + 1} under q",
            sum(p * utility(init_wealth, o) for p, o in zip(prob, gamble)),
        )  # expected utility

    q = get_probabilities(rn_prob)

    print("q")
    print(q)
