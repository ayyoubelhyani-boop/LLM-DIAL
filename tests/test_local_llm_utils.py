from dialin_llm.llm_utils import (
    JsonCache,
    LocalTransformersCoherenceEvaluator,
    _build_coherence_messages,
    _build_coherence_retry_messages,
    _build_naming_messages,
    _extract_label,
    _parse_good_bad_loose,
    _parse_good_bad_loose_or_default_bad,
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


def test_parse_good_bad_loose_or_default_bad_maps_uncertain_output_to_bad() -> None:
    assert _parse_good_bad_loose_or_default_bad("Maybe") == "bad"
    assert _parse_good_bad_loose_or_default_bad("unclear cluster") == "bad"


class FakeGenerator:
    def __init__(self, responses: list[str]) -> None:
        self.model = "fake-model"
        self.responses = responses
        self.calls: list[list[dict[str, str]]] = []

    def generate(self, messages: list[dict[str, str]]) -> str:
        self.calls.append(messages)
        return self.responses.pop(0)


def test_local_coherence_evaluator_retries_after_invalid_output() -> None:
    generator = FakeGenerator(["Maybe", "Bad"])
    evaluator = LocalTransformersCoherenceEvaluator(
        generator=generator,
        cache=JsonCache(),
        prompt_style="benchmark",
    )

    assert evaluator.coherence_eval(["refund my order", "get refund for purchase"]) is False
    assert len(generator.calls) == 2
    assert "previous answer was invalid" in generator.calls[1][1]["content"].lower()


def test_build_coherence_retry_messages_requests_strict_binary_answer() -> None:
    retry_messages = _build_coherence_retry_messages(["refund my order"], prompt_style="benchmark")

    assert "exactly one token" in retry_messages[0]["content"].lower()
    assert "if there is any uncertainty, return bad" in retry_messages[1]["content"].lower()
