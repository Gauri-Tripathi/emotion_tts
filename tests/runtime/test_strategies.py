from unitts.contracts import SynthesisRequest
from unitts.runtime.strategies import AutoregressiveStrategy, FlowMatchingStrategy


def test_ar_compatibility_grouping() -> None:
    strategy = AutoregressiveStrategy()
    base = SynthesisRequest(text="a", engine_id="qwen3-tts", controls={"codec": "12hz", "top_p": 0.9})
    same = SynthesisRequest(text="b", engine_id="qwen3-tts", controls={"codec": "12hz", "top_p": 0.9})
    different = SynthesisRequest(text="c", engine_id="qwen3-tts", controls={"codec": "25hz", "top_p": 0.9})
    assert strategy.compatibility_key(base) == strategy.compatibility_key(same)
    assert strategy.compatibility_key(base) != strategy.compatibility_key(different)


def test_flow_matching_compatibility_grouping() -> None:
    strategy = FlowMatchingStrategy()
    first = SynthesisRequest(text="a", engine_id="f5-tts", controls={"duration_frames": 100, "steps": 32})
    same_bucket = SynthesisRequest(text="b", engine_id="f5-tts", controls={"duration_frames": 200, "steps": 32})
    other_solver = SynthesisRequest(text="c", engine_id="f5-tts", controls={"duration_frames": 200, "steps": 16})
    assert strategy.compatibility_key(first) == strategy.compatibility_key(same_bucket)
    assert strategy.compatibility_key(first) != strategy.compatibility_key(other_solver)
