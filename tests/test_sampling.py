import numpy as np

from dialin_llm.sampling import convex_hull_sample, farthest_first_sample, random_sample, sample_indices


def test_random_sample_is_deterministic_and_capped() -> None:
    indices = [0, 1, 2, 3, 4, 5]
    sample_a = random_sample(indices, sample_size=3, seed=17)
    sample_b = random_sample(indices, sample_size=3, seed=17)

    assert sample_a == sample_b
    assert len(sample_a) == 3


def test_farthest_first_sample_is_deterministic_and_capped() -> None:
    indices = [0, 1, 2, 3]
    embeddings = np.array(
        [
            [0.0, 0.0],
            [0.0, 1.0],
            [1.0, 0.0],
            [5.0, 5.0],
        ]
    )

    sample_a = farthest_first_sample(indices, embeddings=embeddings, sample_size=2, seed=11)
    sample_b = farthest_first_sample(indices, embeddings=embeddings, sample_size=2, seed=11)

    assert sample_a == sample_b
    assert len(sample_a) == 2


def test_convex_hull_sample_is_deterministic_and_prefers_extremes() -> None:
    indices = [0, 1, 2, 3, 4, 5]
    embeddings = np.array(
        [
            [0.0, 0.0],
            [0.0, 2.0],
            [2.0, 2.0],
            [2.0, 0.0],
            [1.0, 1.0],
            [1.1, 1.0],
        ]
    )

    sample_a = convex_hull_sample(indices, embeddings=embeddings, sample_size=4, seed=5)
    sample_b = convex_hull_sample(indices, embeddings=embeddings, sample_size=4, seed=5)

    assert sample_a == sample_b
    assert len(sample_a) == 4
    assert set(sample_a) == {0, 1, 2, 3}


def test_sample_indices_returns_all_when_cluster_is_small() -> None:
    indices = [3, 8]
    embeddings = np.array([[0.0, 0.0], [1.0, 1.0]])

    sample = sample_indices(indices, sample_size=5, sampler="farthest", embeddings=embeddings, seed=3)

    assert sample == indices


def test_sample_indices_supports_convex_sampler() -> None:
    indices = [0, 1, 2, 3, 4]
    embeddings = np.array(
        [
            [0.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
            [1.0, 0.0],
            [0.5, 0.5],
        ]
    )

    sample = sample_indices(indices, sample_size=3, sampler="convex", embeddings=embeddings, seed=9)

    assert len(sample) == 3
    assert 4 not in sample
