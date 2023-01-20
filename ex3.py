# Jonathan Shaki, Or Shachar 204920367, 209493709

from input_parser import build_input_cache
from expectation_maximization import ExpectationMaximization, Expectation, Maximization

from word_counter import word_appearances
from itertools import chain

import matplotlib.pyplot as plt

import numpy as np

INPUT_FILE = "develop.txt"
CLUSTERS_NUMBER = 9
SMALLEST_ITEM_TO_CALC = -15
SMOOTH_LAMBDA = 0.08
SMOOTH_EPSILON = 0.005

USE_WORD_IF_APPEAR_MORE_THAN = 3

STOP_THRESHOLD = 20


def print_confusion_matrix(assignments, items):
    with open('topics.txt') as topics_file:
        topics = topics_file.read().split()
        print(topics)

    matrix = [[0 for _ in range(CLUSTERS_NUMBER + 1)] for _ in range(CLUSTERS_NUMBER)]

    t = 0

    for i, item in enumerate(items):
        assignment = np.argmax(assignments[:, [i]])
        for topic in item.topics:
            matrix[assignment][topics.index(topic)] += 1

        matrix[assignment][-1] += 1

    argsorted = np.argsort(list(-m[-1] for m in matrix))

    print(argsorted)

    for i, item in enumerate(items):
        assignment = np.argmax(assignments[:, [i]])
        if topics[argsorted[assignment]] in item.topics:
            t += 1

    print(t / len(items))

    matrix = sorted(matrix, key=lambda r: -r[-1])

    print(matrix)


def main():
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

    likelihood_history = []
    perplexity_history = []

    maximization()  # starting by maximizing because we already have an assignments
    expectation()
    likelihood = em.calculate_minus_log_likelihood()
    likelihood_history.append(likelihood)
    perplexity_history.append(em.calculate_perplexity())
    while True:  # em.calculate_minus_log_likelihood() < STOP_THRESHOLD:
        maximization()
        expectation()

        new_likelihood = em.calculate_minus_log_likelihood()
        perplexity = em.calculate_perplexity()

        print(f'{new_likelihood}, {perplexity}')

        if likelihood - new_likelihood < STOP_THRESHOLD:
            break

        likelihood = new_likelihood
        likelihood_history.append(likelihood)
        perplexity_history.append(perplexity)

    # plotting graphs

    x_values = list(range(len(likelihood_history)))
    plt.plot(x_values, likelihood_history)
    plt.show()

    plt.plot(x_values, perplexity_history)
    plt.show()

    print_confusion_matrix(em.assignments, items)


if __name__ == "__main__":
    main()
