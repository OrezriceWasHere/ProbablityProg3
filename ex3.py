# Jonathan Shaki, Or Shachar 204920367, 209493709

from input_parser import build_input_cache
from expectation_maximization import ExpectationMaximization, Expectation, Maximization

from word_counter import word_appearances
from itertools import chain

import numpy as np


INPUT_FILE = "develop.txt"
CLUSTERS_NUMBER = 9
SMALLEST_ITEM_TO_CALC = -10
SMOOTH_LAMBDA = 1
SMOOTH_EPSILON = 0.001

USE_WORD_IF_APPEAR_MORE_THAN = 3

STOP_THRESHOLD = 0.5


def main():
    np.set_printoptions(edgeitems=10)

    items = build_input_cache(INPUT_FILE)

    all_words = list(chain.from_iterable(item.words for item in items))
    appearances = word_appearances(all_words)

    clusters = [[] for i in range(CLUSTERS_NUMBER)]
    for index, item in enumerate(items):
        item.words = [word for word in item.words if appearances[word] > USE_WORD_IF_APPEAR_MORE_THAN]
        clusters[index % CLUSTERS_NUMBER].append(item.words)

    docs = [item.words for item in items]

    em = ExpectationMaximization(clusters, docs, CLUSTERS_NUMBER, SMALLEST_ITEM_TO_CALC, SMOOTH_LAMBDA, SMOOTH_EPSILON)
    expectation = Expectation(em)
    maximization = Maximization(em)

    maximization()  # starting by maximizing because we already have an assignments
    expectation()
    while True:  # em.calculate_minus_log_likelihood() < STOP_THRESHOLD:
        print(f'after expectation: {em.calculate_minus_log_likelihood()}, {em.calculate_perplexity()}')
        maximization()
        expectation.update_z_vectors()
        print(f'after maximization: {em.calculate_minus_log_likelihood()}, {em.calculate_perplexity()}')
        expectation()
        expectation.update_z_vectors()


if __name__ == "__main__":
    main()
