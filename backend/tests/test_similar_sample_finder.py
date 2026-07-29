import numpy as np

from services.similar_sample_finder import SimilarSampleFinder


def test_similar_sample_distances_use_global_feature_scale():
    finder = SimilarSampleFinder()
    corpus = np.array([
        [1.0, 10.0, 100.0, 5.0, 2.0],
        [2.0, 20.0, 200.0, 6.0, 3.0],
    ])

    distances = finder._compute_distances(corpus, corpus[0])

    assert distances[0] == 0.0
    assert distances[1] > 0.0
