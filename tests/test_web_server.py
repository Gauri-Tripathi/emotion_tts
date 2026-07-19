from unitts.web_server import provider_payloads


def test_web_provider_payloads_are_available_without_loading_models() -> None:
    providers = provider_payloads()

    qwen = next(provider for provider in providers if provider["name"] == "qwen3-tts")
    assert qwen["capabilities"]["requires_gpu"] is True
