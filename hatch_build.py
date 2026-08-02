"""Build hook: put the HTTP layer back for server installs.

`[tool.hatch.build.targets.wheel] exclude` drops `src/embedx/api` so the PyPI
distribution ships the library alone (task 21). The catch that exclusion alone
does not handle: **`pip install .` builds that same wheel target**, so a source
install -- which is how Docker, the systemd path and `pip install git+...` all
get embedx -- would silently lose the server too, and `embedx serve` would not
exist in the container. Verified, not assumed: installing `.[server]` without
this hook leaves `embedx.api` unimportable.

There is no signal at build time distinguishing "building for PyPI" from
"building for a server install", so it has to be said explicitly. Setting
`EMBEDX_BUILD_SERVER=1` re-includes the package.

The default is the library, deliberately: a wheel published to PyPI cannot be
un-published, whereas a source install that forgets the flag fails loudly and
immediately at `embedx serve`. The safer accident is the recoverable one.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from hatchling.builders.hooks.plugin.interface import BuildHookInterface

#: Set to "1" in the Dockerfile and in the documented source installs.
BUILD_SERVER_ENV_VAR = "EMBEDX_BUILD_SERVER"

_API_SOURCE = Path("src") / "embedx" / "api"


class ServerLayerBuildHook(BuildHookInterface):  # type: ignore[type-arg]
    """Re-include `embedx.api` when the build is for a server install."""

    PLUGIN_NAME = "embedx-server-layer"

    def initialize(self, version: str, build_data: dict[str, Any]) -> None:
        if os.environ.get(BUILD_SERVER_ENV_VAR) != "1":
            return
        root = Path(self.root)
        sources = sorted((root / _API_SOURCE).glob("*.py"))
        if not sources:
            raise RuntimeError(
                f"{BUILD_SERVER_ENV_VAR}=1 was set but {_API_SOURCE} holds no modules; "
                "refusing to build a server wheel with no server in it"
            )
        for source in sources:
            build_data["force_include"][str(source)] = f"embedx/api/{source.name}"
