"""The layer boundaries, enforced.

`data_harness` is layered llm -> core -> data -> app. A layer may import the
ones below it and nothing above. Without a test this is a paragraph in a
README that decays on the first deadline, which is how the package became one
flat namespace in the first place.

The check is static, over the AST, not by importing. Runtime import checks
would pass vacuously here: nearly every heavyweight dependency in this package
is already behind a function-local import, so `import data_harness.core` pulls
in no pandas today even though the source referred to `SessionCache` all over.
Static analysis sees the reference regardless of where it sits.
"""

from __future__ import annotations

import ast
import pathlib

import pytest

PACKAGE = pathlib.Path(__file__).resolve().parent.parent / "data_harness"

#: bottom to top. A layer may import itself and anything earlier.
LAYERS = ("llm", "core", "data", "app")
RANK = {name: index for index, name in enumerate(LAYERS)}

#: Not layered. `eval` is a research harness that sits beside the stack;
#: `__init__` is the package root re-exporting the public surface.
UNLAYERED = {"eval", "__init__"}


def _layer_of(module: str) -> str | None:
    """`data_harness.core.loop` -> `core`. ``None`` for unlayered modules."""
    parts = module.split(".")
    if len(parts) < 2 or parts[0] != "data_harness":
        return None
    return parts[1] if parts[1] in RANK else None


def _module_name(path: pathlib.Path) -> str:
    relative = path.relative_to(PACKAGE.parent).with_suffix("")
    name = ".".join(relative.parts)
    return name.removesuffix(".__init__")


def _imports(path: pathlib.Path) -> set[str]:
    found: set[str] = set()
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, ast.ImportFrom) and node.module:
            if node.module.startswith("data_harness"):
                found.add(node.module)
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.startswith("data_harness"):
                    found.add(alias.name)
    return found


def _source_files() -> list[pathlib.Path]:
    return [p for p in sorted(PACKAGE.rglob("*.py")) if "__pycache__" not in p.parts]


def test_the_package_is_actually_layered():
    """Guards the guard: if the directories vanish, the rest is vacuous."""
    present = {
        p.name for p in PACKAGE.iterdir() if p.is_dir() and p.name != "__pycache__"
    }
    assert set(LAYERS) <= present


@pytest.mark.parametrize("path", _source_files(), ids=_module_name)
def test_module_only_imports_its_own_layer_or_below(path: pathlib.Path):
    """No module may import from a layer above its own.

    `core` importing `data` is the specific violation this was written for:
    the loop used to construct a `SessionCache` and call `format_tool_output`
    directly, which made the harness untestable and unusable without pandas.
    It asks a `RunEnvironment` now.
    """
    module = _module_name(path)
    layer = _layer_of(module)
    if layer is None:
        return

    violations = []
    for imported in sorted(_imports(path)):
        target = _layer_of(imported)
        if target is not None and RANK[target] > RANK[layer]:
            violations.append(f"{layer}/{module} -> {target}/{imported}")

    assert not violations, "layer violation:\n  " + "\n  ".join(violations)


#: Domain names the lower layers must not use in code. Prose is fine and
#: often necessary: explaining *why* the boundary exists means naming what is
#: on the other side of it.
BANNED_IN_CODE = ("SessionCache", "format_tool_output")

#: Third-party packages the lower layers must not require, even lazily. A
#: function-local `import pandas` is still a dependency; it just fails later.
BANNED_DEPENDENCIES = ("pandas", "numpy", "duckdb", "pyarrow")


def _code_identifiers(path: pathlib.Path) -> set[str]:
    """Identifiers referenced in code. Excludes docstrings and comments."""
    names: set[str] = set()
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, ast.Name):
            names.add(node.id)
        elif isinstance(node, ast.Attribute):
            names.add(node.attr)
        elif isinstance(node, ast.alias):
            names.add(node.asname or node.name.split(".")[0])
        elif isinstance(node, ast.arg) and node.annotation is None:
            names.add(node.arg)
    return names


@pytest.mark.parametrize("layer", ["llm", "core"])
def test_lower_layers_do_not_name_the_data_domain(layer: str):
    """The loop must not reach for a `SessionCache`, however indirectly.

    It used to construct one and call `format_tool_output` on it, which is why
    the harness could not be exercised or reused without the data stack. It
    asks a `RunEnvironment` now.
    """
    offenders = []
    for path in _source_files():
        if _layer_of(_module_name(path)) != layer:
            continue
        used = _code_identifiers(path)
        for name in BANNED_IN_CODE:
            if name in used:
                offenders.append(f"{_module_name(path)} uses {name!r}")
    assert not offenders, "\n".join(offenders)


@pytest.mark.parametrize("layer", ["llm", "core"])
def test_lower_layers_do_not_depend_on_the_data_stack(layer: str):
    """No pandas in `llm` or `core`, not even behind a lazy import.

    This is the claim that makes the boundary worth having, so it is checked
    statically: at runtime almost every heavy import in this package is
    function-local, so nothing would fail until the branch was reached.
    """
    offenders = []
    for path in _source_files():
        if _layer_of(_module_name(path)) != layer:
            continue
        for node in ast.walk(ast.parse(path.read_text())):
            modules = []
            if isinstance(node, ast.Import):
                modules = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module:
                modules = [node.module]
            for module in modules:
                if module.split(".")[0] in BANNED_DEPENDENCIES:
                    offenders.append(f"{_module_name(path)} imports {module!r}")
    assert not offenders, "\n".join(offenders)


def test_every_layer_says_what_it_is_for():
    """Each layer package documents its own boundary, next to the code."""
    for layer in LAYERS:
        docstring = ast.get_docstring(
            ast.parse((PACKAGE / layer / "__init__.py").read_text())
        )
        assert docstring, f"{layer} has no docstring"
        assert "import" in docstring.lower(), (
            f"{layer}'s docstring does not state what it may import"
        )
