import numpy as np

from dialin_llm.io import SentenceRecord
from dialin_llm.iterative import run_iterative_clustering


class KeywordEvaluator:
    def coherence_eval(self, sentences: list[str]) -> bool:
        has_refund = ["refund" in sentence for sentence in sentences]
        has_cancel = ["cancel" in sentence for sentence in sentences]
        return all(has_refund) or all(has_cancel)


def test_iterative_loop_removes_sentences_from_remaining_pool() -> None:
    records = [
        SentenceRecord("s1", "refund order please"),
        SentenceRecord("s2", "refund payment issue"),
        SentenceRecord("s3", "cancel my subscription"),
        SentenceRecord("s4", "cancel this plan"),
        SentenceRecord("s5", "weather tomorrow"),
    ]
    embeddings = np.array(
        [
            [1.0, 0.0],
            [1.1, 0.1],
            [0.0, 1.0],
            [0.1, 1.1],
            [4.0, 4.0],
        ]
    )

    result = run_iterative_clustering(
        records,
        embeddings,
        candidate_ks=[1, 2, 3],
        evaluator=KeywordEvaluator(),
        clusterer="kmeans",
        sample_size=5,
        sampler="random",
        epsilon=0.0,
        tmax=3,
        seed=9,
    )

    accepted_ids = {sentence_id for cluster in result.clusters for sentence_id in cluster.member_sentence_ids}

    assert accepted_ids == {"s1", "s2", "s3", "s4"}
    assert result.remaining_sentence_ids == ["s5"]


def test_best_k_is_selected_by_good_over_bad_plus_one_score() -> None:
    records = [
        SentenceRecord("s1", "refund order"),
        SentenceRecord("s2", "refund billing"),
        SentenceRecord("s3", "cancel subscription"),
        SentenceRecord("s4", "cancel membership"),
        SentenceRecord("s5", "weather tomorrow"),
    ]
    embeddings = np.array(
        [
            [1.0, 0.0],
            [1.1, 0.1],
            [0.0, 1.0],
            [0.1, 1.1],
            [4.0, 4.0],
        ]
    )

    result = run_iterative_clustering(
        records,
        embeddings,
        candidate_ks=[1, 3],
        evaluator=KeywordEvaluator(),
        clusterer="kmeans",
        sample_size=5,
        sampler="random",
        epsilon=0.0,
        tmax=1,
        seed=4,
    )

    assert result.iterations[0].selected_k == 3
    scores = {candidate.k: candidate.score for candidate in result.iterations[0].candidate_results}
    assert scores[3] > scores[1]
