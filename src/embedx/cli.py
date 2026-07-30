"""embedx command-line interface.

Task 00 provides only a stub. `serve`/`info`/`check` are implemented in later
tasks (see .claude/tasks/09_cli_and_serve.md). Keep this importable without the GPU
extra.
"""

from __future__ import annotations

import typer

from embedx import __version__

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


@app.command()
def serve() -> None:
    """Start the embedding server (not implemented yet)."""
    typer.echo("embedx serve: not implemented yet")
    raise typer.Exit(code=0)


if __name__ == "__main__":
    app()
