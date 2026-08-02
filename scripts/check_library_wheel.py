"""Assert a built wheel is the LIBRARY wheel, not a server build.

The single source of this check. Both `.github/workflows/ci.yml` (every push)
and `.github/workflows/publish.yml` (every tag, immediately before upload) run
this file rather than carrying their own copy, because two copies of a rule
drift and the one on the publish path is the one that must not.

What it enforces, and why each half matters:

* No `embedx/api/` and no unit file. The wheel published to PyPI is the
  library (task 21): `embedx serve` is registered only when `embedx.api`
  imports, and the systemd unit's ExecStart runs that command. Shipping
  either to PyPI hands someone a server that cannot start.
* The library itself is still there. An `exclude` pattern that matched too
  much would produce a wheel that passes the first half by being empty.

Stdlib only, and no embedx import: it runs on a bare runner python against a
built artifact, before anything is installed.

Usage:  python scripts/check_library_wheel.py [dist-dir]
Exit:   0 if the wheel is the library wheel, 1 with a reason if not.
"""

from __future__ import annotations

import sys
import zipfile
from pathlib import Path

#: Present in a server build, and never in a published wheel.
FORBIDDEN_PREFIXES = ("embedx/api/",)
FORBIDDEN_SUFFIXES = (".service",)

#: A sample of what the library must still contain, so an over-broad exclude
#: cannot pass by removing everything.
REQUIRED = (
    "embedx/__init__.py",
    "embedx/cli.py",
    "embedx/config.py",
    "embedx/registry.py",
    "embedx/engine/engine.py",
    "embedx/engine/scheduling.py",
    "embedx/gpu/vendor.py",
)


def check(dist_dir: Path) -> list[str]:
    """Return a list of problems; empty means the wheel is the library wheel."""
    wheels = sorted(dist_dir.glob("*.whl"))
    if len(wheels) != 1:
        return [f"expected exactly one wheel in {dist_dir}, found {[w.name for w in wheels]}"]

    names = zipfile.ZipFile(wheels[0]).namelist()
    problems = []

    leaked = [n for n in names if n.startswith(FORBIDDEN_PREFIXES)]
    if leaked:
        problems.append(f"the HTTP layer leaked into the wheel: {leaked}")

    units = [n for n in names if n.endswith(FORBIDDEN_SUFFIXES)]
    if units:
        problems.append(f"the systemd unit leaked into the wheel: {units}")

    missing = [n for n in REQUIRED if n not in names]
    if missing:
        problems.append(f"the exclusion took the library with it; missing: {missing}")

    if not problems:
        print(f"{wheels[0].name}: library wheel confirmed ({len(names)} members)")
        print("  no embedx/api/, no unit file, library intact")
    return problems


def main(argv: list[str]) -> int:
    dist_dir = Path(argv[1]) if len(argv) > 1 else Path("dist")
    if not dist_dir.is_dir():
        print(f"no such directory: {dist_dir}", file=sys.stderr)
        return 1
    problems = check(dist_dir)
    for problem in problems:
        print(f"ERROR: {problem}", file=sys.stderr)
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
