from typing import Literal

from fastapi import FastAPI
from pydantic import BaseModel


class HealthResponse(BaseModel):
    status: Literal["ok"]
    service: Literal["executor"]


app = FastAPI(title="Census Insight Executor", version="0.1.0")


@app.get("/health", response_model=HealthResponse, tags=["health"])
def health() -> HealthResponse:
    """Report executor process health."""
    return HealthResponse(status="ok", service="executor")
