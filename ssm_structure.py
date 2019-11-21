import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from py_stringmatching import Levenshtein


def string_similarity(some, other):
    return Levenshtein().get_sim_score(some, other)


def self_similarity_matrix(items, metric):
    return np.array([[metric(x, y) for x in items] for y in items])


def draw_ssm(ssm_lines, representation='string similarity',
             song_name='song name', artist_name='artist name'):
    sns.heatmap(data=ssm_lines)
    plt.show()



