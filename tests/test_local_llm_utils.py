from dialin_llm.llm_utils import (
    _build_coherence_messages,
    _build_naming_messages,
    _extract_label,
    _parse_good_bad_loose,
    _resolve_single_device,
    _uses_sharded_device_map,
)


def test_parse_good_bad_loose_accepts_embedded_verdict() -> None:
    assert _parse_good_bad_loose("The answer is Good.") == "good"
    assert _parse_good_bad_loose("Bad cluster") == "bad"


def test_extract_label_accepts_embedded_label() -> None:
    assert _extract_label("Suggested label: refund-order") == "refund-order"
    assert _extract_label("use payment-refund please") == "payment-refund"
    assert _extract_label("atmmalfunction") == "general-atmmalfunction"
    assert _extract_label("ATM malfunction issue") == "atm-malfunction"


def test_resolve_single_device_respects_cpu_and_cuda_defaults() -> None:
    assert _resolve_single_device(device_map="cpu", cuda_available=True) == "cpu"
    assert _resolve_single_device(device_map="auto", cuda_available=True) == "cuda"
    assert _resolve_single_device(device_map="cuda:1", cuda_available=True) == "cuda:1"
    assert _resolve_single_device(device_map="1", cuda_available=True) == "cuda:1"
    assert _resolve_single_device(device_map="auto", cuda_available=False) == "cpu"


def test_uses_sharded_device_map_only_for_accelerate_modes() -> None:
    assert _uses_sharded_device_map("auto") is True
    assert _uses_sharded_device_map("balanced") is True
    assert _uses_sharded_device_map("cuda:1") is False
    assert _uses_sharded_device_map("cpu") is False


def test_benchmark_prompt_style_adds_examples_and_strict_answer_format() -> None:
    coherence_messages = _build_coherence_messages(["refund my order", "get refund for order"], prompt_style="benchmark")
    naming_messages = _build_naming_messages(["where is my package", "track package status"], prompt_style="benchmark")

    assert "Example A" in coherence_messages[1]["content"]
    assert "Answer:" in coherence_messages[1]["content"]
    assert "verb-object labels" in naming_messages[1]["content"]
    assert "track-package" in naming_messages[1]["content"]
