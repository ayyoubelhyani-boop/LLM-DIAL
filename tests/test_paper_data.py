import json

from dialin_llm.paper_data import _sample_clusters_to_rows, _sample_utterances_to_rows, parse_sample_input


def test_parse_sample_input_extracts_sentence_list() -> None:
    raw = "\ninput:['hello there', 'general kenobi', '']"

    parsed = parse_sample_input(raw)

    assert parsed == ["hello there", "general kenobi"]


def test_sample_clusters_to_rows_preserves_reference_output() -> None:
    payload = [
        {
            "db_id": "Bank",
            "input": "\ninput:['apply for a loan', 'loan application status']",
            "output": "inquire-loan",
        }
    ]

    rows = _sample_clusters_to_rows(payload, "dialin-label")

    assert rows == [
        {
            "sample_id": "label-0",
            "db_id": "Bank",
            "task": "label",
            "reference_output": "inquire-loan",
            "num_sentences": 2,
            "sentences": ["apply for a loan", "loan application status"],
        }
    ]


def test_sample_utterances_to_rows_flattens_sentences() -> None:
    payload = [
        {
            "db_id": "User",
            "input": "\ninput:['first sentence', 'second sentence']",
            "output": "聚类效果好",
        }
    ]

    rows = _sample_utterances_to_rows(payload, "dialin-goodness")

    assert json.loads(json.dumps(rows, ensure_ascii=False)) == [
        {
            "sentence_id": "goodness-0-0",
            "sample_id": "goodness-0",
            "db_id": "User",
            "task": "goodness",
            "reference_output": "聚类效果好",
            "text": "first sentence",
        },
        {
            "sentence_id": "goodness-0-1",
            "sample_id": "goodness-0",
            "db_id": "User",
            "task": "goodness",
            "reference_output": "聚类效果好",
            "text": "second sentence",
        },
    ]
