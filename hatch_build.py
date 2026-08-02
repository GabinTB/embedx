"""Build hook: put the server-side files back for server installs.

`[tool.hatch.build.targets.wheel] exclude` drops the server half so the PyPI
distribution ships the library alone (task 21). The catch that exclusion alone
does not handle: **`pip install .` builds that same wheel target**, so a source
install -- which is how Docker, the systemd path and `pip install git+...` all
get embedx -- would silently lose the server too, and `embedx serve` would not
exist in the container. Verified, not assumed: installing `.[server]` without
this hook leaves `embedx.api` unimportable.

There is no signal at build time distinguishing "building for PyPI" from
"building for a server install", so it has to be said explicitly. Setting
`EMBEDX_BUILD_SERVER=1` re-includes them.

The default is the library, deliberately: a wheel published to PyPI cannot be
un-published, whereas a source install that forgets the flag fails loudly and
immediately at `embedx serve`. The safer accident is the recoverable one.

Two things are server-side, and they travel together on purpose. `embedx.api`
is the HTTP layer. `service/embedx.service` is the systemd unit, whose
`ExecStart` runs `embedx serve` -- a command that does not exist in a library
wheel, so shipping the unit there would hand someone a file that cannot work.
One flag governs both; there is no second mechanism for the unit.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from hatchling.builders.hooks.plugin.interface import BuildHookInterface

#: Set to "1" in the Dockerfile and in the documented source installs.
BUILD_SERVER_ENV_VAR = "EMBEDX_BUILD_SERVER"

_SRC = Path("src") / "embedx"
#: (directory relative to the project root, glob) -> every match is re-included
#: at `embedx/<dir name>/<file name>`. Each entry must also appear in the wheel
#: target's `exclude`, or it would ship unconditionally and the flag would be
#: decorative.
SERVER_SOURCES: tuple[tuple[Path, str], ...] = (
    (_SRC / "api", "*.py"),
    (_SRC / "service", "*.service"),
)


class ServerLayerBuildHook(BuildHookInterface):  # type: ignore[type-arg]
    """Re-include the server-side files when building for a server install."""

    PLUGIN_NAME = "embedx-server-layer"

    def initialize(self, version: str, build_data: dict[str, Any]) -> None:
        if os.environ.get(BUILD_SERVER_ENV_VAR) != "1":
            return
        root = Path(self.root)
        for directory, pattern in SERVER_SOURCES:
            sources = sorted((root / directory).glob(pattern))
            if not sources:
                raise RuntimeError(
                    f"{BUILD_SERVER_ENV_VAR}=1 was set but {directory} holds no "
                    f"{pattern} files; refusing to build a server wheel with a "
                    "piece of the server missing"
                )
            for source in sources:
                build_data["force_include"][str(source)] = f"embedx/{directory.name}/{source.name}"
