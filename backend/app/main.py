from typing import Literal

from fastapi import FastAPI
from pydantic import BaseModel


class HealthResponse(BaseModel):
    status: Literal["ok"]
    service: Literal["backend"]


app = FastAPI(title="Census Insight Agent API", version="0.1.0")


@app.get("/health", response_model=HealthResponse, tags=["health"])
def health() -> HealthResponse:
    """Report process health without contacting external dependencies."""
    return HealthResponse(status="ok", service="backend")
