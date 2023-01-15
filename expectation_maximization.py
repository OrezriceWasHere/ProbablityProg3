import math
import numpy as np
from calculations import softmax
from word_counter import *
from itertools import chain


class ExpectationMaximization:

    def __init__(self,
                 basic_clusters: List[List[List[str]]],
                 docs: List[List[str]],
                 clusters_number,
                 smallest_item_softmax_calc: float,
                 smooth_lambda: float,
                 smooth_epsilon: float):
        self.docs = docs
        self.clusters_number = clusters_number
        self.docs_count = len(docs)
        print(f'number of docs: {self.docs_count}')
        self.clusters_probabilities = np.fromiter([len(cluster) / self.docs_count for cluster in basic_clusters],
                                                  dtype=float)
        self.docs_words = [word_appearances(doc) for doc in docs]
        self.words = set(chain.from_iterable(self.docs))
        print(f'number of words: {len(self.words)}')
        self.word_cluster_probability = {}
        self.expectation = Expectation(self)
        self.maximization = Maximization(self)
        self.smallest_item_softmax_calc = smallest_item_softmax_calc
        self.smooth_lambda = smooth_lambda
        self.smooth_epsilon = smooth_epsilon
        self.assignments = None
        self.z_vectors = None
        self._init_clusters_probabilities()
        self._init_word_cluster_probability()
        self.words_count = sum(len(doc) for doc in self.docs)

    def _init_word_cluster_probability(self):
        # Whenever implemented, weights should rely on output of maximization step
        for cluster in range(self.clusters_number):
            for word in self.words:
                # Even distribution in the beginning TODO: sure? I think it should be based on the first clusters
                self.word_cluster_probability[word] = np.zeros(self.clusters_number)

    def _hot_dot(self, index):
        v = np.zeros(self.clusters_number)
        v[index] = 1
        return v

    def _init_clusters_probabilities(self):
        self.assignments = np.column_stack([self._hot_dot(i % self.clusters_number) for i in range(len(self.docs))])

    def calculate_perplexity(self):
        return np.exp(self.calculate_minus_log_likelihood() / self.words_count)

    def calculate_minus_log_likelihood(self):
        return - sum(np.max(vector) + np.sum(np.exp(vector - np.max(vector))) for vector in self.z_vectors)


class Expectation:

    def __init__(self, em: ExpectationMaximization):
        self.em = em

    def _ln_prob_document_to_be_of_given_cluster(self, document: List[str], cluster_index: int) -> float:
        ln_prob_sum = math.log(self.em.clusters_probabilities[cluster_index])

        for word, count in word_appearances(document).items():
            prob_word = self.em.word_cluster_probability[word][
                cluster_index]
            ln_prob_sum += math.log(prob_word) * count

        return ln_prob_sum

    def z_vector(self, document: List[str]) -> np.ndarray:
        """For each document, a vector of probabilities will be returned, \
        Stating the probabilities for this document to belong to any of the cluster"""

        z_vector = np.fromiter(
            (self._ln_prob_document_to_be_of_given_cluster(document, index)
             for index in range(self.em.clusters_number)), dtype=float
        )
        z_vector[z_vector - np.max(z_vector) < self.em.smallest_item_softmax_calc] = float("-inf")
        return z_vector

    def __call__(self):
        self.em.z_vectors = [self.z_vector(doc) for doc in self.em.docs]
        self.em.assignments = np.column_stack([softmax(vector - np.max(vector)) for vector in self.em.z_vectors])


class Maximization:

    def __init__(self, em: ExpectationMaximization):
        self.em = em

    def __call__(self):

        # Update cluster weights
        # Update word cluster probability

        self.em.clusters_probabilities = (self.em.assignments.sum(axis=0) + self.em.smooth_epsilon) / (
                1 + self.em.clusters_number * self.em.smooth_epsilon)

        for word in self.em.word_cluster_probability.keys():
            for i in range(self.em.clusters_number):
                self.em.word_cluster_probability[word][i] = (sum(
                    self.em.assignments[i][j] * self.em.docs_words[i].get(word, 0) for j in
                    range(len(self.em.docs))) + self.em.smooth_lambda) / (sum(
                    self.em.assignments[i][j] * len(self.em.docs[i]) for j in
                    range(len(self.em.docs))) + self.em.smooth_lambda * len(self.em.words))
