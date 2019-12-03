import numpy as np
from py_stringmatching import Levenshtein


def string_similarity(some, other):
    return Levenshtein().get_sim_score(some, other)


def self_similarity_matrix(items, metric):
    return np.array([[metric(x, y) for x in items] for y in items])

def headline_from(header):
    return header + '\n' + ''.join(['-' for _ in range(len(header))])