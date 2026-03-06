from dialin_llm.iterative import IntentCluster
from dialin_llm.merge import merge_clusters_by_label


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
