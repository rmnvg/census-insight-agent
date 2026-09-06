import os
from dataclasses import dataclass


@dataclass(frozen=True)
class UiSettings:
    api_base_url: str = "http://backend:8000"
    connect_timeout_seconds: float = 5.0
    read_timeout_seconds: float = 180.0
    max_artifact_bytes: int = 10_485_760

    @classmethod
    def from_environment(cls) -> "UiSettings":
        return cls(
            api_base_url=os.getenv("CENSUS_API_BASE_URL", cls.api_base_url).rstrip("/"),
            connect_timeout_seconds=float(
                os.getenv("CENSUS_UI_CONNECT_TIMEOUT_SECONDS", cls.connect_timeout_seconds)
            ),
            read_timeout_seconds=float(
                os.getenv("CENSUS_UI_READ_TIMEOUT_SECONDS", cls.read_timeout_seconds)
            ),
            max_artifact_bytes=int(
                os.getenv("CENSUS_UI_MAX_ARTIFACT_BYTES", cls.max_artifact_bytes)
            ),
        )
