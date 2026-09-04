import re

from backend.app.config import Settings


class ProviderError(RuntimeError):
    """Base error for provider operations safe to show to an operator."""


class EmbeddingInputError(ProviderError):
    """The caller supplied invalid embedding input."""


class EmbeddingResponseError(ProviderError):
    """The provider returned an invalid embedding response."""


class ProviderRequestError(ProviderError):
    """A provider request failed after sensitive details were removed."""


_SENSITIVE_PATTERNS = (
    re.compile(r"(?i)(authorization:\s*bearer\s+)[^\s,;]+"),
    re.compile(r"(?i)(access[_ -]?token[=:]\s*)[^\s,;]+"),
)


def sanitize_provider_error(error: BaseException, settings: Settings) -> str:
    """Return useful provider details without credential paths or bearer tokens."""
    message = str(error)
    sensitive_values = (
        settings.google_application_credentials,
        settings.google_credentials_host_path,
    )
    for value in sensitive_values:
        if value:
            message = message.replace(value, "[REDACTED]")
    for pattern in _SENSITIVE_PATTERNS:
        message = pattern.sub(r"\1[REDACTED]", message)
    return message
