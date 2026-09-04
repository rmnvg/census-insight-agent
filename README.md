# census-insight-agent

Initial service scaffold for a census insight agent. See [docs/DECISIONS.md](docs/DECISIONS.md) for the fixed architecture decisions.

## Development

Requires Python 3.12 and `uv`.

```shell
cp .env.example .env
uv sync --frozen
make check
```

Run the services locally with the `run-backend`, `run-frontend`, and `run-executor` Make targets, or use Docker Compose after setting `GOOGLE_CREDENTIALS_HOST_PATH` to an ADC JSON file.
