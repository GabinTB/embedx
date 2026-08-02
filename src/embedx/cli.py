"""embedx command-line interface: serve, info, check.

Importable without the GPU extra — torch enters only when a model is
actually loaded, which no command does at startup any more. `serve` builds
an empty registry; `check --warm` is the one path that loads on purpose.

Importable without the HTTP layer, too. The PyPI wheel ships the library and
excludes `embedx.api`, so `serve` is registered only when that package is
actually present (see `_api_available`). This module must therefore never
import `embedx.api` at module level: the console script is `embedx.cli:app`,
so a module-level import would make a library install fail on `--help` and
`info` as well, not just on the one command that needs it.

Precedence note: `Settings` treats init kwargs as highest priority
(CLI > env > file > defaults), so an option the user did not pass must
never reach `Settings`. Every option below defaults to the `None`
sentinel and `_make_settings` forwards only explicit values; forwarding
typer defaults would silently override `EMBEDX_*` and the config file.
"""

from __future__ import annotations

import importlib.util
import logging
from typing import Annotated, Any

import typer
from pydantic import ValidationError
from pydantic_settings import SettingsError

from embedx import __version__
from embedx.config import Pooling, Settings
from embedx.gpu.budgets import device_budgets
from embedx.gpu.discovery import discover_devices, rank_devices
from embedx.gpu.vendor import get_accelerator
from embedx.registry import ModelRegistry, RegistryError

logger = logging.getLogger("embedx.cli")

VALID_LOG_LEVELS = ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL")

# Said in `--help` rather than raised at runtime. Someone who came looking for
# `serve` and does not find it should learn where the server went, not
# conclude the project has none.
_NO_SERVER_EPILOG = (
    "This install is the library only, so `serve` is not available. The HTTP "
    "server ships with the Docker image or a source install "
    "(github.com/GabinTB/embedx); everything else here works as documented."
)


def _api_available() -> bool:
    """Whether the HTTP layer was installed alongside the library.

    `find_spec` rather than a try/except import: this runs at import time on
    every CLI invocation, and importing `embedx.api` would pull in fastapi
    just to decide whether to offer one subcommand.
    """
    return importlib.util.find_spec("embedx.api") is not None


_HAS_API = _api_available()

app = typer.Typer(
    name="embedx",
    help="Heterogeneous multi-GPU text embedding server.",
    epilog=None if _HAS_API else _NO_SERVER_EPILOG,
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
RevisionOpt = Annotated[str | None, typer.Option(help="Model revision (branch/tag/commit).")]
NormalizeOpt = Annotated[
    bool | None, typer.Option("--normalize/--no-normalize", help="L2-normalize embeddings.")
]
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
DefaultKeepAliveOpt = Annotated[
    float | None, typer.Option(help="Seconds an idle model stays resident.")
]
MaxLoadedModelsOpt = Annotated[
    int | None, typer.Option(help="Max models resident at once; unset means no cap.")
]
WarmOpt = Annotated[
    str | None,
    typer.Option(help="Preflight a model id end to end: load it, then unload it."),
]
WarmPoolingOpt = Annotated[
    Pooling | None, typer.Option(help="Pooling for --warm. Required with it; never inferred.")
]


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


def _ranked_devices(settings: Settings) -> Any:
    """Discover and rank devices, failing the way a preflight should."""
    try:
        infos = discover_devices(settings.devices)
    except ValueError as exc:  # a requested device index does not exist
        typer.echo(f"check failed: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    if not infos:
        vendor = get_accelerator().name.upper()
        typer.echo(
            f"check failed: no {vendor} device available (torch missing or {vendor} unavailable)",
            err=True,
        )
        raise typer.Exit(code=1)
    return rank_devices(infos, settings.device_weights)


def _build_registry(settings: Settings, ranked: Any) -> ModelRegistry:
    """Separate function so tests can substitute a stubbed registry."""
    return ModelRegistry(ranked, settings=settings)


def _run_uvicorn(application: Any, settings: Settings) -> None:
    """Separate function so tests can monkeypatch the actual server start."""
    import uvicorn

    uvicorn.run(
        application,
        host=settings.host,
        port=settings.port,
        log_level=settings.log_level.lower(),
    )


def serve(
    revision: RevisionOpt = None,
    normalize: NormalizeOpt = None,
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
    default_keep_alive_s: DefaultKeepAliveOpt = None,
    max_loaded_models: MaxLoadedModelsOpt = None,
) -> None:
    """Start the embedding server. Loads no model until a request names one."""
    settings = _make_settings(dict(locals()))
    _configure_logging(settings)

    # Nothing is loaded here. Devices are still enumerated up front, because
    # a box with no usable GPU should fail at startup rather than on the
    # first request.
    ranked = _ranked_devices(settings)
    registry = _build_registry(settings, ranked)
    registry.start()
    # Imported here, not at module level, for two reasons. It must stay out of
    # `locals()` above, which is forwarded verbatim to `Settings` and would
    # reject an unknown key; and this module has to import on a library
    # install where `embedx.api` is absent (see the module docstring).
    from embedx.api import create_app

    application = create_app(settings, registry)

    logger.info("binding http://%s:%d", settings.host, settings.port)
    logger.info("api key: %s", "set" if settings.api_key else "not set (open access)")
    logger.info(
        "normalize=%s wrapping=%r default_keep_alive_s=%s max_loaded_models=%s",
        settings.normalize,
        settings.wrapping,
        settings.default_keep_alive_s,
        settings.max_loaded_models if settings.max_loaded_models is not None else "unlimited",
    )
    budgets = device_budgets(ranked, settings.max_batch_tokens, settings.device_batch_tokens)
    for device in ranked:
        logger.info(
            "device %d: %s (weight %.3f, max_batch_tokens %d)",
            device.index,
            device.name,
            device.weight,
            budgets[device.index],
        )
    logger.info("no models loaded; each is loaded on the first request that names it")
    if settings.is_exposed_without_auth:
        logger.warning(settings.exposure_warning())

    try:
        _run_uvicorn(application, settings)
    finally:
        registry.stop()
        registry.evict_all()


# Registered only when the HTTP layer is installed. On a library-only wheel
# `serve` is therefore absent from `--help` rather than present and failing,
# which is why no runtime error for the missing case exists anywhere below.
# Registration happens here rather than at the end of the module so the
# command order in `--help` stays serve, info, check.
if _HAS_API:
    app.command()(serve)


@app.command()
def info(
    revision: RevisionOpt = None,
    normalize: NormalizeOpt = None,
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
    default_keep_alive_s: DefaultKeepAliveOpt = None,
    max_loaded_models: MaxLoadedModelsOpt = None,
) -> None:
    """Print the resolved configuration and device table. Loads no model."""
    settings = _make_settings(dict(locals()))
    _configure_logging(settings)

    typer.echo(f"revision:          {settings.revision or '-'}")
    typer.echo(f"normalize:         {settings.normalize}")
    typer.echo(f"wrapping:          {settings.wrapping or '-'}")
    typer.echo(f"bind:              {settings.host}:{settings.port}")
    typer.echo(f"api_key:           {'set' if settings.api_key else 'not set'}")
    typer.echo(f"max_batch_tokens:  {settings.max_batch_tokens}")
    typer.echo(f"max_batch_items:   {settings.max_batch_items or '-'}")
    typer.echo(f"max_request_items: {settings.max_request_items}")
    typer.echo(f"max_request_bytes: {settings.max_request_bytes}")
    typer.echo(f"default_keep_alive_s: {settings.default_keep_alive_s}")
    typer.echo(f"max_loaded_models: {settings.max_loaded_models or 'unlimited'}")
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
        vendor = get_accelerator().name.upper()
        typer.echo(f"devices: none visible (torch not installed or {vendor} unavailable)")
        raise typer.Exit(code=0)

    ranked = rank_devices(infos, settings.device_weights)
    budgets = device_budgets(ranked, settings.max_batch_tokens, settings.device_batch_tokens)
    typer.echo("devices (ranked fastest first):")
    for device in ranked:
        typer.echo(
            f"  [{device.index}] {device.name}  weight={device.weight:.3f}  "
            f"max_batch_tokens={budgets[device.index]}"
        )

    # This process is not the server. Its registry was constructed a moment
    # ago and has loaded nothing, so an empty list here says nothing at all
    # about what a running embedx is holding -- say so, rather than letting
    # the output be read as a status report on the service.
    loaded = _build_registry(settings, ranked).list_loaded()
    typer.echo("")
    typer.echo("models loaded IN THIS PROCESS (not the running server; ask it via GET /info):")
    if not loaded:
        typer.echo("  none - this command loads nothing")
    for status in loaded:
        typer.echo(
            f"  {status.model_id}  pooling={status.pooling.value}  "
            f"dtype={status.dtype.value}  devices={list(status.device_indices)}  "
            f"idle={status.idle_s:.0f}s  refs={status.ref_count}"
        )


@app.command()
def check(
    revision: RevisionOpt = None,
    normalize: NormalizeOpt = None,
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
    default_keep_alive_s: DefaultKeepAliveOpt = None,
    max_loaded_models: MaxLoadedModelsOpt = None,
    warm: WarmOpt = None,
    warm_pooling: WarmPoolingOpt = None,
) -> None:
    """Preflight (for systemd): config parses, requested devices exist.

    No model is loaded unless `--warm` asks for one, because there is no
    configured model to preflight any more.
    """
    values = dict(locals())
    # Preflight-only flags, not configuration: they must not reach Settings,
    # which forbids unknown fields.
    values.pop("warm", None)
    values.pop("warm_pooling", None)
    settings = _make_settings(values)
    _configure_logging(settings)

    if warm is not None and warm_pooling is None:
        typer.echo(
            "check failed: --warm needs --warm-pooling. Pooling is never inferred, "
            "here or anywhere else: the wrong one loads without error and returns "
            f"plausible but wrong vectors. Pass one of: "
            f"{', '.join(member.value for member in Pooling)}",
            err=True,
        )
        raise typer.Exit(code=2)
    if warm is None and warm_pooling is not None:
        typer.echo("check failed: --warm-pooling only means something with --warm", err=True)
        raise typer.Exit(code=2)

    ranked = _ranked_devices(settings)

    # An exposure warning is a warning, not a failure.
    if settings.is_exposed_without_auth:
        typer.echo(f"warning: {settings.exposure_warning()}")

    if warm is not None:
        assert warm_pooling is not None  # guarded above
        registry = _build_registry(settings, ranked)
        try:
            # keep_alive=0 makes this a load-then-evict by construction:
            # the registry drops a model with keep_alive<=0 the moment the
            # last reference goes, so nothing is left resident and the
            # cleanup cannot be forgotten here.
            with registry.acquire(warm, pooling=warm_pooling, keep_alive=0) as engine:
                engine.embed(["embedx preflight"])
        except RegistryError as exc:
            typer.echo(f"check failed: could not load {warm}: {exc}", err=True)
            raise typer.Exit(code=1) from exc
        except Exception as exc:  # a broken checkpoint, a bad path, OOM
            typer.echo(f"check failed: could not load {warm}: {exc}", err=True)
            raise typer.Exit(code=1) from exc
        resident = registry.list_loaded()
        if resident:  # the keep_alive=0 contract did not hold
            registry.evict_all()
            typer.echo(f"check failed: {warm} was still resident after the warm cycle", err=True)
            raise typer.Exit(code=1)
        typer.echo(f"warm ok: {warm} loaded and unloaded, nothing left resident")

    typer.echo(f"check ok: config valid, {len(ranked)} device(s) available")


if __name__ == "__main__":
    app()
