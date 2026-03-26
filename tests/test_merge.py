from dialin_llm.iterative import IntentCluster
from dialin_llm.merge import merge_clusters_by_label, name_clusters


def test_merge_clusters_by_label_merges_close_labels() -> None:
    clusters = [
        IntentCluster(
            cluster_id="cluster-0",
            member_sentence_ids=["s1", "s2"],
            sentences=["refund my order", "refund the purchase"],
            iteration_found=1,
            label="refund-order",
            source_cluster_ids=["cluster-0"],
        ),
        IntentCluster(
            cluster_id="cluster-1",
            member_sentence_ids=["s3"],
            sentences=["refund an order item"],
            iteration_found=1,
            label="refund-orders",
            source_cluster_ids=["cluster-1"],
        ),
        IntentCluster(
            cluster_id="cluster-2",
            member_sentence_ids=["s4"],
            sentences=["cancel shipping request"],
            iteration_found=1,
            label="cancel-shipping",
            source_cluster_ids=["cluster-2"],
        ),
    ]

    merged = merge_clusters_by_label(clusters, theta=1.1, seed=5)

    assert len(merged) == 2
    assert merged[0].member_sentence_ids == ["s1", "s2", "s3"]
    assert merged[0].source_cluster_ids == ["cluster-0", "cluster-1"]


def test_hybrid_merge_can_merge_semantically_close_clusters_with_different_labels() -> None:
    clusters = [
        IntentCluster(
            cluster_id="cluster-0",
            member_sentence_ids=["s1", "s2"],
            sentences=["refund my order", "refund the purchase"],
            iteration_found=1,
            label="refund-order",
            source_cluster_ids=["cluster-0"],
        ),
        IntentCluster(
            cluster_id="cluster-1",
            member_sentence_ids=["s3", "s4"],
            sentences=["need money back for purchase", "how to get money back on my order"],
            iteration_found=1,
            label="money-back-request",
            source_cluster_ids=["cluster-1"],
        ),
    ]

    label_only = merge_clusters_by_label(clusters, theta=1.5, strategy="label", seed=5)
    hybrid = merge_clusters_by_label(clusters, theta=1.5, strategy="hybrid", label_weight=0.2, seed=5)

    assert len(label_only) == 2
    assert len(hybrid) == 1


def test_merge_rejects_invalid_label_weight() -> None:
    clusters = [
        IntentCluster(
            cluster_id="cluster-0",
            member_sentence_ids=["s1"],
            sentences=["refund order"],
            iteration_found=1,
            label="refund-order",
            source_cluster_ids=["cluster-0"],
        ),
        IntentCluster(
            cluster_id="cluster-1",
            member_sentence_ids=["s2"],
            sentences=["refund purchase"],
            iteration_found=1,
            label="refund-purchase",
            source_cluster_ids=["cluster-1"],
        ),
    ]

    try:
        merge_clusters_by_label(clusters, strategy="hybrid", label_weight=1.5)
    except ValueError as exc:
        assert "label_weight" in str(exc)
    else:
        raise AssertionError("Expected ValueError for invalid label_weight")


class RecordingNamer:
    def __init__(self) -> None:
        self.calls: list[list[str]] = []

    def name_cluster(self, sentences: list[str]) -> str:
        self.calls.append(list(sentences))
        return "refund-order"


def test_name_clusters_can_use_centroid_sampler() -> None:
    namer = RecordingNamer()
    clusters = [
        IntentCluster(
            cluster_id="cluster-0",
            member_sentence_ids=["s1", "s2", "s3", "s4"],
            sentences=[
                "refund order",
                "refund order please",
                "please refund order",
                "track package",
            ],
            iteration_found=1,
            source_cluster_ids=["cluster-0"],
        )
    ]

    named = name_clusters(clusters, namer, sample_size=2, sampler="centroid")

    assert named[0].label == "refund-order"
    assert len(namer.calls) == 1
    assert len(namer.calls[0]) == 2
    assert all("refund" in sentence for sentence in namer.calls[0])
