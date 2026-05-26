"""Static pre-flight validation of pipeline stage chains (Workstream A).

Walks stages forward over a set of "available field tokens" (see the field
vocabulary in the plan/spec) and reports broken chains BEFORE execution.
Pure, dependency-free set logic — no type checking, no audio, no models.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

from voxkitchen.pipeline.spec import IngestSpec, PipelineSpec


def _namespace(token: str) -> str | None:
    """Return the namespace prefix for a wildcard token like 'metrics.*'."""
    return token[:-2] if token.endswith(".*") else None


def is_satisfied(required: str, available: set[str]) -> bool:
    """True if ``required`` is met by ``available`` (exact or namespace match)."""
    if required in available:
        return True
    # required is specific (e.g. metrics.snr); a wildcard 'metrics.*' satisfies it
    if "." in required:
        prefix = required.rsplit(".", 1)[0]
        if f"{prefix}.*" in available:
            return True
    # required is itself a wildcard 'metrics.*'; any 'metrics.<k>' satisfies it
    ns = _namespace(required)
    if ns is not None:
        return any(a == ns or a.startswith(f"{ns}.") for a in available)
    return False


def apply_writes(available: set[str], writes: list[str]) -> set[str]:
    """Return a new available set with ``writes`` added."""
    return available | set(writes)


def apply_clears(available: set[str], clears: list[str]) -> set[str]:
    """Return a new available set with ``clears`` removed.

    Clearing a wildcard 'metrics.*' removes every 'metrics.<k>' token.
    """
    out = set(available)
    for token in clears:
        ns = _namespace(token)
        if ns is not None:
            out = {a for a in out if not (a == ns or a.startswith(f"{ns}."))}
        else:
            out.discard(token)
    return out


_SUPERVISION_FIELDS = (
    "supervisions.text",
    "supervisions.language",
    "supervisions.speaker",
    "supervisions.gender",
)


def _initial_available(ingest: IngestSpec) -> set[str]:
    """Field tokens present before any stage runs, based on the ingest source.

    ``dir`` scans bare audio files. ``manifest`` and ``recipe`` load CutSets
    that commonly already carry transcripts/speaker/language, so we assume the
    supervision fields MAY be present — pre-flight prefers a false negative
    (missing a real gap) over a false positive (blocking a valid pipeline).
    """
    available = {"audio"}
    if ingest.source in ("manifest", "recipe"):
        available.update(_SUPERVISION_FIELDS)
    if ingest.source == "dir" and ingest.args.get("reference_text_glob"):
        available.add("custom.reference_text")
    return available


# ---------------------------------------------------------------------------
# Forward-walk preflight — checks operator field contracts across all stages
# ---------------------------------------------------------------------------


@dataclass
class PreflightResult:
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


@dataclass
class _Contract:
    reads: list[str]
    writes: list[str]
    optional_reads: list[str]
    clears: list[str]


def _contract_from_registry(stage_op: str, args: dict[str, object]) -> _Contract | None:
    """Return the contract for an importable operator, or None if not importable."""
    from voxkitchen.operators.registry import UnknownOperatorError, get_operator

    try:
        op_cls = get_operator(stage_op)
    except UnknownOperatorError:
        return None
    dynamic: list[str] = []
    try:
        cfg = op_cls.config_cls.model_validate(args)
        dynamic = op_cls(cfg, ctx=None).dynamic_reads()  # type: ignore[arg-type]
    except Exception:
        # Bad args (e.g. a malformed filter condition) are reported by the
        # arg-validation pass and at runtime; here we degrade to static reads
        # rather than crash the pre-flight check.
        dynamic = []
    return _Contract(
        reads=list(op_cls.reads) + dynamic,
        writes=list(op_cls.writes),
        optional_reads=list(op_cls.optional_reads),
        clears=list(op_cls.clears),
    )


def preflight_spec(
    spec: PipelineSpec,
    *,
    contract_lookup: Callable[[str, dict[str, object]], _Contract | None] = _contract_from_registry,
) -> PreflightResult:
    """Walk stages forward; report broken chains as errors/warnings.

    Operators with no resolvable contract (out-of-env, or unknown) are skipped
    conservatively: we stop tracking after them so we never raise false alarms
    about fields a stage we couldn't inspect might have produced.
    """
    result = PreflightResult()
    available: set[str] = _initial_available(spec.ingest)
    tracking = True

    for stage in spec.stages:
        contract = contract_lookup(stage.op, stage.args)
        if contract is None:
            tracking = False
            continue
        if tracking:
            for token in contract.reads:
                if not is_satisfied(token, available):
                    result.errors.append(
                        f"stage {stage.name!r} (op {stage.op!r}) requires {token!r} "
                        f"but no upstream stage produces it"
                    )
            for token in contract.optional_reads:
                if not is_satisfied(token, available):
                    result.warnings.append(
                        f"stage {stage.name!r} (op {stage.op!r}) can use {token!r} "
                        f"but no upstream stage produces it (stage will skip/degrade)"
                    )
            available = apply_clears(available, contract.clears)
            available = apply_writes(available, contract.writes)

    return result


def contract_from_schemas(
    stage_op: str, args: dict[str, object], schemas: dict[str, object]
) -> _Contract | None:
    """Build a contract from an op_schemas.json entry (out-of-env operators)."""
    info = schemas.get(stage_op)
    if not info or not isinstance(info, dict) or "contract" not in info:
        return None
    c = info["contract"]
    if not isinstance(c, dict):
        return None
    return _Contract(
        reads=list(c.get("reads", [])),
        writes=list(c.get("writes", [])),
        optional_reads=list(c.get("optional_reads", [])),
        clears=list(c.get("clears", [])),
    )


def make_contract_lookup(
    schemas: dict[str, object] | None,
) -> Callable[[str, dict[str, object]], _Contract | None]:
    """Registry first (gets dynamic_reads); op_schemas.json fallback otherwise.

    Note: dynamic_reads is intentionally not carried on the schema-fallback path.
    This is safe because the only operator that uses dynamic_reads is
    quality_score_filter, which has no required_extras and is always importable
    in-env, so it is always resolved via the registry path first.
    """

    def lookup(stage_op: str, args: dict[str, object]) -> _Contract | None:
        c = _contract_from_registry(stage_op, args)
        if c is not None:
            return c
        if schemas is not None:
            return contract_from_schemas(stage_op, args, schemas)
        return None

    return lookup
