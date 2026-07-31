"""embedx command-line interface: serve, info, check.

Importable without the GPU extra — torch enters only through
`build_engine`, which `serve` alone calls.

Precedence note: `Settings` treats init kwargs as highest priority
(CLI > env > file > defaults), so an option the user did not pass must
never reach `Settings`. Every option below defaults to the `None`
sentinel and `_make_settings` forwards only explicit values; forwarding
typer defaults would silently override `EMBEDX_*` and the config file.
"""

from __future__ import annotations

import logging
from typing import Annotated, Any

import typer
from pydantic import ValidationError
from pydantic_settings import SettingsError

from embedx import __version__
from embedx.api import create_app
from embedx.backend.factory import build_engine
from embedx.config import Dtype, Pooling, Settings
from embedx.gpu.budgets import device_budgets
from embedx.gpu.discovery import discover_devices, rank_devices

logger = logging.getLogger("embedx.cli")

VALID_LOG_LEVELS = ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL")

app = typer.Typer(
    name="embedx",
    help="Heterogeneous multi-GPU text embedding server.",
    no_args_is_help=True,
    add_completion=False,
)


def _version_callback(value: bool) -> None:
    if value:
        typer.echo(__version__)
        raise typer.Exit()


@app.callback()
def main(
    _version: bool = typer.Option(
        False,
        "--version",
        callback=_version_callback,
        is_eager=True,
        help="Show the embedx version and exit.",
    ),
) -> None:
    """embedx CLI entry point."""


# Shared option aliases; every default is the None sentinel (see module
# docstring for why that is load-bearing).
ModelIdOpt = Annotated[str | None, typer.Option(help="Hugging Face model id or local path.")]
RevisionOpt = Annotated[str | None, typer.Option(help="Model revision (branch/tag/commit).")]
PoolingOpt = Annotated[
    Pooling | None, typer.Option(help="Pooling strategy (required, no default).")
]
NormalizeOpt = Annotated[
    bool | None, typer.Option("--normalize/--no-normalize", help="L2-normalize embeddings.")
]
DtypeOpt = Annotated[Dtype | None, typer.Option(help="Compute dtype.")]
MaxSeqLenOpt = Annotated[int | None, typer.Option(help="Max sequence length (tokens).")]
HostOpt = Annotated[str | None, typer.Option(help="Bind address.")]
PortOpt = Annotated[int | None, typer.Option(help="Bind port.")]
ApiKeyOpt = Annotated[str | None, typer.Option(help="Bearer API key; unset disables auth.")]
LogLevelOpt = Annotated[str | None, typer.Option(help="Log level (DEBUG..CRITICAL).")]
DevicesOpt = Annotated[str | None, typer.Option(help='CUDA device indices, e.g. "0,1".')]
MaxBatchTokensOpt = Annotated[int | None, typer.Option(help="Default per-batch token budget.")]
MaxBatchItemsOpt = Annotated[int | None, typer.Option(help="Per-batch item cap.")]
DeviceWeightsOpt = Annotated[
    str | None, typer.Option(help='Weight overrides, e.g. "0=1.0,1=0.35".')
]
DeviceBatchTokensOpt = Annotated[
    str | None, typer.Option(help='Budget overrides, e.g. "0=16384,1=4096".')
]
MaxRequestItemsOpt = Annotated[int | None, typer.Option(help="Max inputs per request.")]
MaxRequestBytesOpt = Annotated[int | None, typer.Option(help="Max request body bytes.")]


def _make_settings(values: dict[str, Any]) -> Settings:
    provided = {key: value for key, value in values.items() if value is not None}
    try:
        return Settings(**provided)
    except (ValidationError, SettingsError) as exc:
        typer.echo(f"configuration error: {exc}", err=True)
        raise typer.Exit(code=2) from exc


def _configure_logging(settings: Settings) -> None:
    level = settings.log_level.upper()
    if level not in VALID_LOG_LEVELS:
        typer.echo(
            f"invalid log_level {settings.log_level!r}: valid levels are "
            f"{', '.join(VALID_LOG_LEVELS)}",
            err=True,
        )
        raise typer.Exit(code=2)
    # No force=True: if a handler already exists (tests, embedding callers)
    # basicConfig is a no-op and only the level is applied.
    logging.basicConfig(
        level=getattr(logging, level),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    logging.getLogger().setLevel(getattr(logging, level))


def _run_uvicorn(application: Any, settings: Settings) -> None:
    """Separate function so tests can monkeypatch the actual server start."""
    import uvicorn

    uvicorn.run(
        application,
        host=settings.host,
        port=settings.port,
        log_level=settings.log_level.lower(),
    )


@app.command()
def serve(
    model_id: ModelIdOpt = None,
    revision: RevisionOpt = None,
    pooling: PoolingOpt = None,
    normalize: NormalizeOpt = None,
    dtype: DtypeOpt = None,
    max_seq_len: MaxSeqLenOpt = None,
    host: HostOpt = None,
    port: PortOpt = None,
    api_key: ApiKeyOpt = None,
    log_level: LogLevelOpt = None,
    devices: DevicesOpt = None,
    max_batch_tokens: MaxBatchTokensOpt = None,
    max_batch_items: MaxBatchItemsOpt = None,
    device_weights: DeviceWeightsOpt = None,
    device_batch_tokens: DeviceBatchTokensOpt = None,
    max_request_items: MaxRequestItemsOpt = None,
    max_request_bytes: MaxRequestBytesOpt = None,
) -> None:
    """Start the embedding server."""
    settings = _make_settings(dict(locals()))
    _configure_logging(settings)

    engine = build_engine(settings)
    application = create_app(settings, engine)

    logger.info("binding http://%s:%d", settings.host, settings.port)
    logger.info("api key: %s", "set" if settings.api_key else "not set (open access)")
    logger.info(
        "pooling=%s dtype=%s normalize=%s",
        settings.pooling.value,
        settings.dtype.value,
        settings.normalize,
    )
    for device, budget in engine.devices_with_budgets:
        logger.info(
            "device %d: %s (weight %.3f, max_batch_tokens %d)",
            device.index,
            device.name,
            device.weight,
            budget,
        )
    if settings.is_exposed_without_auth:
        logger.warning(settings.exposure_warning())

    _run_uvicorn(application, settings)


@app.command()
def info(
    model_id: ModelIdOpt = None,
    revision: RevisionOpt = None,
    pooling: PoolingOpt = None,
    normalize: NormalizeOpt = None,
    dtype: DtypeOpt = None,
    max_seq_len: MaxSeqLenOpt = None,
    host: HostOpt = None,
    port: PortOpt = None,
    api_key: ApiKeyOpt = None,
    log_level: LogLevelOpt = None,
    devices: DevicesOpt = None,
    max_batch_tokens: MaxBatchTokensOpt = None,
    max_batch_items: MaxBatchItemsOpt = None,
    device_weights: DeviceWeightsOpt = None,
    device_batch_tokens: DeviceBatchTokensOpt = None,
    max_request_items: MaxRequestItemsOpt = None,
    max_request_bytes: MaxRequestBytesOpt = None,
) -> None:
    """Print the resolved configuration and device table. Loads no model."""
    settings = _make_settings(dict(locals()))
    _configure_logging(settings)

    typer.echo(f"model_id:          {settings.model_id}")
    typer.echo(f"revision:          {settings.revision or '-'}")
    typer.echo(f"pooling:           {settings.pooling.value}")
    typer.echo(f"normalize:         {settings.normalize}")
    typer.echo(f"dtype:             {settings.dtype.value}")
    typer.echo(f"max_seq_len:       {settings.max_seq_len or '-'}")
    typer.echo(f"bind:              {settings.host}:{settings.port}")
    typer.echo(f"api_key:           {'set' if settings.api_key else 'not set'}")
    typer.echo(f"max_batch_tokens:  {settings.max_batch_tokens}")
    typer.echo(f"max_batch_items:   {settings.max_batch_items or '-'}")
    typer.echo(f"max_request_items: {settings.max_request_items}")
    typer.echo(f"max_request_bytes: {settings.max_request_bytes}")
    if settings.is_exposed_without_auth:
        typer.echo(f"warning: {settings.exposure_warning()}")

    # info is a configuration inspector, not a health check: device problems
    # are reported plainly and still exit 0. `check` is the enforcer.
    try:
        infos = discover_devices(settings.devices)
    except ValueError as exc:
        typer.echo(f"devices: {exc}")
        raise typer.Exit(code=0) from exc
    if not infos:
        typer.echo("devices: none visible (torch not installed or CUDA unavailable)")
        raise typer.Exit(code=0)

    ranked = rank_devices(infos, settings.device_weights)
    budgets = device_budgets(ranked, settings.max_batch_tokens, settings.device_batch_tokens)
    typer.echo("devices (ranked fastest first):")
    for device in ranked:
        typer.echo(
            f"  [{device.index}] {device.name}  weight={device.weight:.3f}  "
            f"max_batch_tokens={budgets[device.index]}"
        )


@app.command()
def check(
    model_id: ModelIdOpt = None,
    revision: RevisionOpt = None,
    pooling: PoolingOpt = None,
    normalize: NormalizeOpt = None,
    dtype: DtypeOpt = None,
    max_seq_len: MaxSeqLenOpt = None,
    host: HostOpt = None,
    port: PortOpt = None,
    api_key: ApiKeyOpt = None,
    log_level: LogLevelOpt = None,
    devices: DevicesOpt = None,
    max_batch_tokens: MaxBatchTokensOpt = None,
    max_batch_items: MaxBatchItemsOpt = None,
    device_weights: DeviceWeightsOpt = None,
    device_batch_tokens: DeviceBatchTokensOpt = None,
    max_request_items: MaxRequestItemsOpt = None,
    max_request_bytes: MaxRequestBytesOpt = None,
) -> None:
    """Preflight (for systemd): config parses, requested devices exist."""
    settings = _make_settings(dict(locals()))
    _configure_logging(settings)

    try:
        infos = discover_devices(settings.devices)
    except ValueError as exc:  # a requested device index does not exist
        typer.echo(f"check failed: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    if not infos:
        typer.echo(
            "check failed: no CUDA device available (torch missing or CUDA unavailable)",
            err=True,
        )
        raise typer.Exit(code=1)

    # An exposure warning is a warning, not a failure.
    if settings.is_exposed_without_auth:
        typer.echo(f"warning: {settings.exposure_warning()}")
    typer.echo(f"check ok: config valid, {len(infos)} device(s) available")


if __name__ == "__main__":
    app()
