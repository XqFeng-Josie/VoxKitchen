"""Process-global operator registry.

Plan 2 ships a manual registry populated via ``register_operator``. Plan 8
will layer entry-points discovery on top via ``voxkitchen.plugins.discovery``
without changing the API here.
"""

from __future__ import annotations

import difflib

from voxkitchen.operators.base import Operator

_REGISTRY: dict[str, type[Operator]] = {}


class UnknownOperatorError(KeyError):
    """Raised when an operator name is not in the registry."""

    def __init__(self, name: str, suggestions: list[str]) -> None:
        self.op_name = name
        self.suggestions = suggestions
        hint = f"; did you mean {', '.join(suggestions)}?" if suggestions else ""
        super().__init__(f"unknown operator: {name!r}{hint}")


class MissingExtrasError(ImportError):
    """Raised when an operator's required extras are not installed."""

    def __init__(self, op_name: str, extras: list[str]) -> None:
        self.op_name = op_name
        self.extras = extras
        extras_str = ",".join(extras)
        super().__init__(
            f"operator {op_name!r} requires extras not installed. "
            f"Install with: pip install voxkitchen[{extras_str}]"
        )


def register_operator(op_cls: type[Operator]) -> type[Operator]:
    """Register an Operator subclass in the global registry.

    Used as a decorator:

        @register_operator
        class MyOp(Operator):
            ...

    Raises ``ValueError`` if the class does not declare a non-empty ``name`` or
    if an operator with the same name is already registered.
    """
    if not getattr(op_cls, "name", ""):
        raise ValueError(f"{op_cls.__name__} must declare a non-empty class-level 'name' attribute")
    if op_cls.name in _REGISTRY:
        existing = _REGISTRY[op_cls.name]
        raise ValueError(
            f"operator {op_cls.name!r} already registered "
            f"(existing: {existing.__module__}.{existing.__name__}, "
            f"new: {op_cls.__module__}.{op_cls.__name__})"
        )
    _REGISTRY[op_cls.name] = op_cls
    return op_cls


def get_operator(name: str) -> type[Operator]:
    """Return the Operator subclass registered under ``name``.

    Raises ``UnknownOperatorError`` with fuzzy-match suggestions if not found.
    """
    from voxkitchen.plugins.discovery import load_plugins

    load_plugins()
    if name in _REGISTRY:
        return _REGISTRY[name]
    suggestions = difflib.get_close_matches(name, list(_REGISTRY.keys()), n=3, cutoff=0.6)
    raise UnknownOperatorError(name, suggestions)


def list_operators() -> list[str]:
    """Return all registered operator names, sorted."""
    return sorted(_REGISTRY.keys())
