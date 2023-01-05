import math
import numpy as np
from calculations import softmax
from word_counter import *


class ExpectationMaximization:

    def __init__(self, basic_clusters: List[List[List[str]]],
                 smallest_item_softmax_calc: float,
                 smooth_lambda: float,
                 vocabulary_count: int):
        self.clusters = basic_clusters
        self.cluster_count = len(self.clusters)
        self.docs_count = sum((len(cluster) for cluster in self.clusters))
        self.cluster_weight = [len(cluster) / self.docs_count for cluster in self.clusters]
        self.word_appearances = [word_appearances_in_cluster(cluster) for cluster in self.clusters]
        self.word_cluster_probability = {}
        self.expectation = Expectation(self)
        self.maximization = Maximization(self)
        self.smallest_item_softmax_calc = smallest_item_softmax_calc
        self.smooth_lambda = smooth_lambda
        self.vocabulary_count = vocabulary_count
        self.init_word_cluster_probability()

    def init_word_cluster_probability(self):
        # Whenever implemented, weights should rely on output of maximization step
        for cluster in range(self.cluster_count):
            for word in self.word_appearances[cluster].keys():
                # Even distribution in the beginning
                self.word_cluster_probability[word] = np.ones(self.cluster_count) / self.cluster_count


class Expectation:

    def __init__(self, em: ExpectationMaximization):
        self.em = em

    def ln_prob_document_to_be_of_given_cluster(self, document: List[str], cluster_index: int) -> float:
        ln_prob_sum = math.log(self.em.cluster_weight[cluster_index])

        for word, count in word_appearances(document).items():
            prob_word = self.em.word_cluster_probability[word][cluster_index] * count
            ln_prob_sum += math.log(prob_word)

        return ln_prob_sum

    def __call__(self, document: List[str]) -> np.ndarray:
        """For each document, a vector of probabilities will be returned, \
        Stating the probabilities for this document to belong to any of the cluster"""

        z_vector = np.fromiter(
            (self.ln_prob_document_to_be_of_given_cluster(document, index)
             for index in range(len(self.em.clusters))), dtype=float
        )
        z_vector = z_vector / z_vector.sum()
        z_vector[z_vector - np.max(z_vector) < self.em.smallest_item_softmax_calc] = float("-inf")
        return softmax(z_vector)


class Maximization:

    def __init__(self, em: ExpectationMaximization):
        self.em = em

    def __call__(self, *args, **kwargs):
        # TODO:
        # Update cluster weights
        # Update word cluster probability
        # Return improvement
        pass
