from backend.app.config import Settings
from backend.app.providers.errors import sanitize_provider_error


def test_provider_error_redacts_credentials() -> None:
    settings = Settings.model_validate(
        {
            "google_cloud_project": "test-project",
            "gemini_chat_model": "test-chat-model",
            "gemini_embedding_model": "test-embedding-model",
            "google_application_credentials": "/private/container-adc.json",
            "google_credentials_host_path": "/private/host-adc.json",
        }
    )
    error = RuntimeError(
        "request failed Authorization: Bearer secret-token "
        "/private/container-adc.json /private/host-adc.json"
    )

    serialized = sanitize_provider_error(error, settings)

    assert "secret-token" not in serialized
    assert "container-adc.json" not in serialized
    assert "host-adc.json" not in serialized
    assert serialized.count("[REDACTED]") == 3
