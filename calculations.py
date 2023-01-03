# Jonathan Shaki, Or Shachar 204920367, 209493709

from typing import List, Tuple
import math


def calculate_perplexity_repetitive_items(probabilities: List[Tuple[int, float]]) -> float:
    for count, probability in probabilities:
        if probability == 0:
            return math.inf  # worst perplexity possible

    sum_logs = sum(math.log(p) * count for count, p in
                   probabilities)  # calculate the logs in order to sum them and avoid underflow
    return math.pow(math.e, -sum_logs / sum(c for c, _ in probabilities))
