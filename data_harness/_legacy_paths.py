"""Keep the pre-layering module paths importable.

Modules moved into `llm`, `core`, `data`, and `app` when the package grew
layers. `data_harness.loop` and friends were documented and in use, so they
stay resolvable.

These are aliases, not copies: `sys.modules["data_harness.loop"]` *is*
`data_harness.core.loop`, so ``isinstance`` checks and identity comparisons
across the two paths agree. Re-exporting names into a shim module would give
the same names but a second module object, and a class imported through one
path would not be the class imported through the other.

New code should use the layered path. The alias exists so a rename does not
become an upgrade barrier.
"""

from __future__ import annotations

import importlib
import importlib.abc
import importlib.machinery
import importlib.util
import sys

#: legacy dotted path -> the module that now implements it
LEGACY_PATHS: dict[str, str] = {
    "data_harness.types": "data_harness.llm.types",
    "data_harness.streaming": "data_harness.llm.streaming",
    "data_harness.testing": "data_harness.llm.testing",
    "data_harness.providers": "data_harness.llm.providers",
    "data_harness.providers.base": "data_harness.llm.providers.base",
    "data_harness.providers.openai": "data_harness.llm.providers.openai",
    "data_harness.providers.anthropic": "data_harness.llm.providers.anthropic",
    "data_harness.loop": "data_harness.data.harness",
    "data_harness.result": "data_harness.core.result",
    "data_harness.artifacts": "data_harness.core.artifacts",
    "data_harness.exceptions": "data_harness.core.exceptions",
    "data_harness.observe": "data_harness.core.observe",
    "data_harness.logger": "data_harness.core.logger",
    "data_harness.schema": "data_harness.core.schema",
    "data_harness.serialize": "data_harness.core.serialize",
    "data_harness.cache": "data_harness.data.cache",
    "data_harness.format": "data_harness.data.format",
    "data_harness.io": "data_harness.data.io",
    "data_harness.exec_cache": "data_harness.data.exec_cache",
    "data_harness.mcp": "data_harness.data.mcp",
    "data_harness.tools": "data_harness.data.tools",
    "data_harness.tools.interpreter": "data_harness.data.tools.interpreter",
    "data_harness.tools.sandbox": "data_harness.data.tools.sandbox",
    "data_harness.tools.sql": "data_harness.data.tools.sql",
    "data_harness.tools.variables": "data_harness.data.tools.variables",
    "data_harness.tools.connectors": "data_harness.data.tools.connectors",
    "data_harness.tools.planner": "data_harness.data.tools.planner",
    "data_harness.tools.subagent": "data_harness.data.tools.subagent",
    "data_harness.agent": "data_harness.app.agent",
    "data_harness.quickstart": "data_harness.app.quickstart",
    "data_harness.cli": "data_harness.app.cli",
    "data_harness.notebook": "data_harness.app.notebook",
    "data_harness.pandas": "data_harness.app.pandas",
}


class _AliasLoader(importlib.abc.Loader):
    """Bind a legacy name to an already-imported module.

    `create_module` hands back the real module and `exec_module` does nothing,
    because the source has already run under its own name. Returning the
    target's own spec instead would make the import machinery build a second
    module object and execute the file again: the two names would then hold
    separate classes, and `isinstance` across them would quietly be false.
    """

    def __init__(self, target: str) -> None:
        self._target = target

    def create_module(self, spec: importlib.machinery.ModuleSpec):
        return importlib.import_module(self._target)

    def exec_module(self, module: object) -> None:
        return None


class _LegacyPathFinder(importlib.abc.MetaPathFinder):
    """Resolve a legacy path to the module that replaced it, on first import.

    A meta-path finder rather than eager aliasing: importing every module at
    package import time would drag the whole library, and every optional
    dependency sitting behind a lazy import, into memory just to register
    names nobody may use.
    """

    def find_spec(
        self, fullname: str, path: object = None, target: object = None
    ) -> importlib.machinery.ModuleSpec | None:
        real = LEGACY_PATHS.get(fullname)
        if real is None:
            return None
        return importlib.util.spec_from_loader(fullname, _AliasLoader(real))


def install() -> None:
    """Make the legacy paths importable. Idempotent.

    Inserted at the front of `sys.meta_path`, not appended. An aliased package
    such as `data_harness.providers` resolves to a real package whose
    ``__path__`` still contains `base.py`, so the default `PathFinder` can
    resolve `data_harness.providers.base` by itself and would load a second
    copy of it. Going first means the alias wins for the names it owns; every
    other name falls through on a dict miss.
    """
    if not any(isinstance(f, _LegacyPathFinder) for f in sys.meta_path):
        sys.meta_path.insert(0, _LegacyPathFinder())
