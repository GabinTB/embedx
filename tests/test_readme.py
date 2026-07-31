"""The README configuration table is rendered from Settings and must match.

`render_config_table()` is the single source of the table; if a Settings
field, default, or description changes without the README, this fails.
Regenerate with:

    uv run python -c \\
      "from tests.test_readme import render_config_table; print(render_config_table())"
"""

from __future__ import annotations

import types
import typing
from enum import Enum
from pathlib import Path

from embedx.config import Settings

README = Path(__file__).resolve().parent.parent / "README.md"


def _type_label(annotation: object) -> str:
    origin = typing.get_origin(annotation)
    if origin in (typing.Union, types.UnionType):
        return " or ".join(_type_label(arg) for arg in typing.get_args(annotation))
    if annotation is type(None):
        return "None"
    if isinstance(annotation, type) and issubclass(annotation, Enum):
        return " / ".join(member.value for member in annotation)
    if origin is not None:
        args = ", ".join(_type_label(arg) for arg in typing.get_args(annotation))
        return f"{origin.__name__}[{args}]"
    return getattr(annotation, "__name__", str(annotation))


def _default_label(field: object) -> str:
    if field.is_required():  # type: ignore[attr-defined]
        return "**required**"
    if field.default_factory is not None:  # type: ignore[attr-defined]
        return repr(field.default_factory())  # type: ignore[attr-defined]
    default = field.default  # type: ignore[attr-defined]
    if isinstance(default, Enum):
        return repr(default.value)
    return repr(default)


def render_config_table() -> str:
    lines = [
        "| Setting | Type | Default | Meaning |",
        "| --- | --- | --- | --- |",
    ]
    for name, field in Settings.model_fields.items():
        lines.append(
            f"| `EMBEDX_{name.upper()}` "
            f"| {_type_label(field.annotation)} "
            f"| {_default_label(field)} "
            f"| {field.description} |"
        )
    return "\n".join(lines)


def test_every_setting_has_a_description() -> None:
    missing = [name for name, field in Settings.model_fields.items() if not field.description]
    assert not missing, f"Settings fields without description: {missing}"


def test_readme_config_table_matches_settings() -> None:
    table = render_config_table()
    assert table in README.read_text(), (
        "README config table has drifted from Settings; regenerate it with:\n"
        '  uv run python -c "from tests.test_readme import render_config_table; '
        'print(render_config_table())"'
    )
