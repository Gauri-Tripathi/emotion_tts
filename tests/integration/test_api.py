import time

from fastapi.testclient import TestClient

from unitts.services.api import create_app


def test_api_smoke_and_dummy_generation() -> None:
    with TestClient(create_app()) as client:
        assert client.get("/health").json() == {"status": "ok"}
        response = client.post("/v1/speech/generations", json={"text": "hello", "engine_id": "dummy"})
        assert response.status_code == 202
        job_id = response.json()["job_id"]
        for _ in range(100):
            payload = client.get(f"/v1/speech/generations/{job_id}").json()
            if payload["status"] == "succeeded":
                break
            time.sleep(0.01)
        assert payload["status"] == "succeeded"
        assert payload["result"]["audio_base64"]
