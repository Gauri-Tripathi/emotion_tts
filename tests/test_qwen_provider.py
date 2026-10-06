import pytest

from unitts import TTSRequest
from unitts.providers.open_weights import Qwen3TTSProvider


class _CustomVoiceModel:
    def __init__(self) -> None:
        self.calls: list[dict[str, str]] = []

    def generate_custom_voice(self, **kwargs: str) -> tuple[list[object], int]:
        self.calls.append(kwargs)
        return [object()], 24000


def test_qwen_custom_voice_uses_supported_default_speaker_and_language_name() -> None:
    provider = Qwen3TTSProvider(device="cpu")
    model = _CustomVoiceModel()
    provider._model = model

    provider._call_inference(TTSRequest(text="Hello", language="en-US"))

    assert model.calls == [{"text": "Hello", "speaker": "ryan", "language": "english"}]


def test_qwen_custom_voice_rejects_voice_cloning_before_model_call() -> None:
    provider = Qwen3TTSProvider(device="cpu")
    provider._model = _CustomVoiceModel()

    with pytest.raises(ValueError, match="does not support voice cloning"):
        provider._call_inference(TTSRequest(text="Hello", reference_audio="reference.wav"))


def test_qwen_custom_voice_rejects_voice_design_before_model_call() -> None:
    provider = Qwen3TTSProvider(device="cpu")
    provider._model = _CustomVoiceModel()

    with pytest.raises(ValueError, match="does not support description-driven"):
        provider._call_inference(TTSRequest(text="Hello", voice_description="warm voice"))


@pytest.mark.parametrize("device,dtype_name,placement", [("cpu", "float32", "cpu"), ("cuda", "float16", "cuda:0")])
def test_qwen_load_places_wrapper_on_requested_device(monkeypatch, device, dtype_name, placement):
    import sys
    from types import SimpleNamespace
    from unittest.mock import Mock
    torch = SimpleNamespace(float16=object(), bfloat16=object(), float32=object())
    monkeypatch.setitem(sys.modules, "torch", torch)

    wrapper = object()  # The real Qwen wrapper has no .to() method.
    load = Mock(return_value=wrapper)
    monkeypatch.setitem(sys.modules, "qwen_tts", SimpleNamespace(Qwen3TTSModel=SimpleNamespace(from_pretrained=load)))
    provider = Qwen3TTSProvider(device=device)
    assert provider._load_model() is wrapper
    load.assert_called_once_with(provider._DEFAULT_MODEL, device_map=placement,
                                 dtype=getattr(torch, dtype_name), attn_implementation="sdpa")


def test_qwen_precision_override(monkeypatch):
    import sys
    from types import SimpleNamespace
    from unittest.mock import Mock
    torch = SimpleNamespace(float16=object(), bfloat16=object(), float32=object())
    monkeypatch.setitem(sys.modules, "torch", torch)

    load = Mock(return_value=object())
    monkeypatch.setitem(sys.modules, "qwen_tts", SimpleNamespace(Qwen3TTSModel=SimpleNamespace(from_pretrained=load)))
    Qwen3TTSProvider(device="cuda", dtype="bfloat16", attn_implementation="eager")._load_model()
    assert load.call_args.kwargs["dtype"] == torch.bfloat16
    assert load.call_args.kwargs["attn_implementation"] == "eager"
