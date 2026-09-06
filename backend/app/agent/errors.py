from collections.abc import Mapping


class AgentOperationalError(RuntimeError):
    def __init__(
        self,
        *,
        code: str,
        node: str,
        message: str,
        retryable: bool,
        elapsed_seconds: float | None,
        configured_timeout_seconds: float | None,
        retry_count: int,
        diagnostics: Mapping[str, object] | None = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.node = node
        self.message = message
        self.retryable = retryable
        self.elapsed_seconds = elapsed_seconds
        self.configured_timeout_seconds = configured_timeout_seconds
        self.retry_count = retry_count
        self.diagnostics = dict(diagnostics or {})
