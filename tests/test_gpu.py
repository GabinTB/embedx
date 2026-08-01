"""Tests for GPU discovery, ranking, and budgets (task 06)."""

from __future__ import annotations

import ast
from collections.abc import Iterator
from pathlib import Path

import pytest

from embedx.gpu import ARCH_FACTOR, DeviceInfo, device_budgets, rank_devices

GIB = 2**30


def make_device(
    index: int,
    sms: int = 100,
    major: int = 8,
    minor: int = 0,
    memory: int = 16 * GIB,
) -> DeviceInfo:
    return DeviceInfo(
        index=index,
        name=f"Fake GPU {index}",
        total_memory_bytes=memory,
        multi_processor_count=sms,
        capability=(major, minor),
    )


# --------------------------------------------------------------------------- #
# rank_devices
# --------------------------------------------------------------------------- #


def test_more_sms_wins_at_equal_capability() -> None:
    ranked = rank_devices([make_device(0, sms=50), make_device(1, sms=100)], {})
    assert [d.index for d in ranked] == [1, 0]
    assert ranked[0].weight == 1.0
    assert ranked[1].weight == pytest.approx(0.5)


def test_newer_capability_wins_at_equal_sm_count() -> None:
    ranked = rank_devices([make_device(0, major=7), make_device(1, major=9)], {})
    assert [d.index for d in ranked] == [1, 0]
    assert ranked[0].score == pytest.approx(100 * ARCH_FACTOR[9])
    assert ranked[1].score == pytest.approx(100 * ARCH_FACTOR[7])


def test_ties_break_by_index_ascending() -> None:
    ranked = rank_devices([make_device(2), make_device(0), make_device(1)], {})
    assert [d.index for d in ranked] == [0, 1, 2]


def test_weight_override_demotes_a_faster_device() -> None:
    # Device 0 outscores device 1 on SMs, but the user says it sits on a
    # narrow PCIe link: the override must reorder the ranking.
    infos = [make_device(0, sms=128), make_device(1, sms=64)]
    assert [d.index for d in rank_devices(infos, {})] == [0, 1]
    ranked = rank_devices(infos, {0: 0.1})
    assert [d.index for d in ranked] == [1, 0]
    assert ranked[1].weight == 0.1  # override used verbatim
    assert ranked[0].weight == pytest.approx(0.5)  # computed, normalised


def test_rank_does_not_mutate_inputs() -> None:
    infos = [make_device(0), make_device(1, sms=50)]
    rank_devices(infos, {})
    assert infos[0].score == 0.0
    assert infos[0].weight == 1.0


def test_unknown_major_falls_back_to_nearest_lower() -> None:
    known = rank_devices([make_device(0, major=10)], {})[0]
    unknown = rank_devices([make_device(0, major=11)], {})[0]
    assert unknown.score == pytest.approx(known.score)
    # Below every known major: lowest known factor, not 1.0.
    ancient = rank_devices([make_device(0, major=3)], {})[0]
    oldest_known = rank_devices([make_device(0, major=min(ARCH_FACTOR))], {})[0]
    assert ancient.score == pytest.approx(oldest_known.score)


def test_rank_empty_input() -> None:
    assert rank_devices([], {}) == []


# --------------------------------------------------------------------------- #
# device_budgets
# --------------------------------------------------------------------------- #


def test_budgets_scale_with_memory() -> None:
    ranked = [make_device(0, memory=16 * GIB), make_device(1, memory=4 * GIB)]
    assert device_budgets(ranked, 16384, {}) == {0: 16384, 1: 4096}


def test_budget_override_bypasses_scaling() -> None:
    ranked = [make_device(0, memory=16 * GIB), make_device(1, memory=4 * GIB)]
    assert device_budgets(ranked, 16384, {1: 9999}) == {0: 16384, 1: 9999}


def test_min_tokens_floors_a_tiny_device() -> None:
    ranked = [make_device(0, memory=16 * GIB), make_device(1, memory=128 * 2**20)]
    budgets = device_budgets(ranked, 16384, {})
    assert budgets[1] == 512  # scaled value would be 128


def test_budgets_empty_input() -> None:
    assert device_budgets([], 16384, {}) == {}


def test_budgets_validation() -> None:
    with pytest.raises(ValueError, match="default_tokens"):
        device_budgets([make_device(0)], 0, {})
    with pytest.raises(ValueError, match="min_tokens"):
        device_budgets([make_device(0)], 16384, {}, min_tokens=0)


# --------------------------------------------------------------------------- #
# The durable no-torch-at-import and no-vendor-leak rules
# --------------------------------------------------------------------------- #

# The one module allowed to know which GPU vendor this is.
VENDOR_MODULE = ("gpu", "vendor.py")


def _is_type_checking_test(test: ast.expr) -> bool:
    return (isinstance(test, ast.Name) and test.id == "TYPE_CHECKING") or (
        isinstance(test, ast.Attribute) and test.attr == "TYPE_CHECKING"
    )


def _module_scope_nodes(tree: ast.Module) -> Iterator[ast.AST]:
    """Every node at module scope, descending into try/if/with blocks but
    never into function bodies (imports there are lazy by definition) and
    never into `if TYPE_CHECKING:` bodies (annotation-only, never executed
    at runtime — the sanctioned home for torch types in annotations)."""
    stack: list[ast.AST] = list(tree.body)
    while stack:
        node = stack.pop()
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
            continue
        if isinstance(node, ast.If) and _is_type_checking_test(node.test):
            stack.extend(node.orelse)  # the else branch does execute
            continue
        yield node
        stack.extend(ast.iter_child_nodes(node))


def test_no_module_level_torch_import_anywhere() -> None:
    src_root = Path(__file__).resolve().parent.parent / "src" / "embedx"
    assert src_root.is_dir()
    offenders = []
    for path in sorted(src_root.rglob("*.py")):
        tree = ast.parse(path.read_text(), filename=str(path))
        for node in _module_scope_nodes(tree):
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
                if any(name == "torch" or name.startswith("torch.") for name in names):
                    offenders.append(f"{path}:{node.lineno}")
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ""
                if module == "torch" or module.startswith("torch."):
                    offenders.append(f"{path}:{node.lineno}")
    assert not offenders, f"module-level torch imports found: {offenders}"


def _is_vendor_module(path: Path) -> bool:
    return path.parts[-2:] == VENDOR_MODULE


def _names_a_vendor(node: ast.AST) -> bool:
    """True for the two shapes a leaked CUDA call takes.

    Attribute access `<anything>.cuda` catches `torch.cuda.empty_cache()`
    and — the reason this is not a grep for the literal string
    "torch.cuda" — the aliased spelling `torch_mod.cuda.get_device_capability()`
    that hf.py used, which no such grep would have found.

    A `"cuda"` / `"cuda:0"` string constant catches the other half: the
    device string. `torch.device("cuda", index)` names no attribute at all,
    so the attribute rule alone would let it straight back in. The f-string
    form `f"cuda:{i}"` is caught too, because its constant piece is `cuda:`.
    Longer prose that merely mentions CUDA (`"CUDA device indices"`) is not
    matched: this is about device strings, not documentation.
    """
    if isinstance(node, ast.Attribute) and node.attr == "cuda":
        return True
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        text = node.value.lower()
        return text == "cuda" or text.startswith("cuda:")
    return False


def test_no_vendor_specific_calls_outside_the_seam() -> None:
    """No module but `gpu/vendor.py` may name a GPU vendor.

    The durable form of the task-18 result. Without it the seam rots: the
    next feature that needs `empty_cache` reaches for `torch.cuda` directly,
    it works, review waves it through, and two commits later the vendor
    surface is spread across five files again.
    """
    src_root = Path(__file__).resolve().parent.parent / "src" / "embedx"
    assert src_root.is_dir()
    offenders = []
    for path in sorted(src_root.rglob("*.py")):
        if _is_vendor_module(path):
            continue
        tree = ast.parse(path.read_text(), filename=str(path))
        for node in ast.walk(tree):  # every scope, not just module level
            if _names_a_vendor(node):
                offenders.append(f"{path.relative_to(src_root)}:{node.lineno}")
    assert not offenders, (
        "vendor-specific references outside gpu/vendor.py: "
        f"{offenders} — go through the Accelerator instead"
    )


def test_the_guard_would_catch_a_reintroduced_leak() -> None:
    """The guard fails on the exact code this task removed.

    A guard nobody has seen fail is a guard nobody knows works — and both
    of these spellings were live in the tree before this commit.
    """
    leaks = [
        "torch.cuda.empty_cache()",  # registry.py
        "major, _ = torch_mod.cuda.get_device_capability(index)",  # hf.py, aliased
        'self._device = torch_mod.device("cuda", device_index)',  # hf.py, device string
        'device = f"cuda:{index}"',  # the f-string spelling
    ]
    for leak in leaks:
        tree = ast.parse(leak)
        assert any(_names_a_vendor(node) for node in ast.walk(tree)), leak

    # And does not fire on prose that merely mentions the vendor, which is
    # why `cli.py`'s 'CUDA device indices' help text is allowed to stay.
    allowed = ast.parse('help_text = "CUDA device indices, e.g. \\"0,1\\"."')
    assert not any(_names_a_vendor(node) for node in ast.walk(allowed))
