"""
GEMMA-AETHER: Atmospheric Evaluation & Tensor Forecasting Orbit
Core Scientific Schemas, 5D Tensor Invariants, Canonical RRF Fusion,
Storm-Event Segmentation, VerticalEncodedConvLSTM Specification,
and 8-Rule Deterministic Provenance Verifier.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple, Any
import math
import re


class ProvenanceClass(str, Enum):
    DIRECT_OBSERVATION = "DIRECT_OBSERVATION"
    RADAR_DERIVED_ESTIMATE = "RADAR_DERIVED_ESTIMATE"
    MODEL_DERIVED_CURRENT = "MODEL_DERIVED_CURRENT"
    REANALYSIS_FIELD = "REANALYSIS_FIELD"
    MODEL_FORECAST = "MODEL_FORECAST"


class ImplementationStatus(str, Enum):
    IMPLEMENTED = "IMPLEMENTED"
    IN_DEVELOPMENT = "IN_DEVELOPMENT"
    PLANNED = "PLANNED"


class ScientificClaimTier(str, Enum):
    TIER_A_FUNCTIONAL = "TIER_A_FUNCTIONAL"
    TIER_B_TECHNICAL_VALIDATION = "TIER_B_TECHNICAL_VALIDATION"
    TIER_C_CONTROLLED_EVALUATION = "TIER_C_CONTROLLED_EVALUATION"
    TIER_D_DEMONSTRATED_FINDING = "TIER_D_DEMONSTRATED_FINDING"


# Canonical L = 7 Isobaric Pressure Levels (hPa) spanning 0.11 km to 11.78 km MSL
CANONICAL_PRESSURE_LEVELS_HPA: Tuple[int, ...] = (1000, 925, 850, 700, 500, 300, 200)

# Canonical T = 13 Discrete Timestamps spanning 12 elapsed intervals
CANONICAL_TIMESTEPS_T: int = 13
CANONICAL_GRID_H: int = 32
CANONICAL_GRID_W: int = 32
CANONICAL_UPPER_CHANNELS_C: int = 7
CANONICAL_SURFACE_CHANNELS_C: int = 10
CANONICAL_FORECAST_HORIZONS_MIN: Tuple[int, ...] = (15, 30, 60, 120, 180)


class CrossSpaceComparisonError(ValueError):
    """Raised when attempting direct cosine similarity between unaligned embedding spaces."""
    pass


def format_canonical_record_id(
    domain_id: str,
    utc_t0_iso: str,
    window_minutes: int = 180,
    cadence_minutes: int = 15,
) -> str:
    """
    Deterministic canonical atmospheric record ID shared across Index A, Index B, and Index C.
    Format: aether_{domain_id}_{utc_t0_compact}_w{window_minutes}m_dt{cadence_minutes}m
    """
    compact_ts = (
        utc_t0_iso.replace("-", "")
        .replace(":", "")
        .replace(".000Z", "Z")
        .split(".")[0]
    )
    if not compact_ts.endswith("Z"):
        compact_ts += "Z"
    return f"aether_{domain_id}_{compact_ts}_w{window_minutes}m_dt{cadence_minutes}m"


def build_temporal_window_timestamps(
    t0_epoch_sec: int,
    cadence_minutes: int = 15,
    num_observations_t: int = CANONICAL_TIMESTEPS_T,
) -> List[int]:
    """
    Constructs T = 13 discrete timestamps spanning (T - 1) = 12 elapsed intervals:
    [t0 - 12*dt, t0 - 11*dt, ..., t0 - dt, t0].
    """
    if num_observations_t < 2:
        raise ValueError("num_observations_t must be >= 2")
    dt_sec = cadence_minutes * 60
    return [
        t0_epoch_sec - (num_observations_t - 1 - idx) * dt_sec
        for idx in range(num_observations_t)
    ]


def verify_cross_space_guard(model_a: str, model_b: str, vec_a: List[float], vec_b: List[float]) -> float:
    """
    Enforces that EmbeddingGemma 2 and gemini-embedding-2 vectors are never directly compared
    via cosine similarity even though both have dimension 768.
    """
    if model_a != model_b:
        raise CrossSpaceComparisonError(
            f"PROHIBITED CROSS-SPACE COMPARISON: Cannot compute direct cosine similarity "
            f"between independently learned spaces '{model_a}' (dim={len(vec_a)}) and "
            f"'{model_b}' (dim={len(vec_b)}). Equal dimensionality does not imply geometric alignment."
        )
    dot = sum(a * b for a, b in zip(vec_a, vec_b))
    norm_a = math.sqrt(sum(a * a for a in vec_a)) or 1.0
    norm_b = math.sqrt(sum(b * b for b in vec_b)) or 1.0
    return dot / (norm_a * norm_b)


def reciprocal_rank_fusion(
    ranked_lists: Dict[str, List[str]],
    k: int = 60,
    weights: Optional[Dict[str, float]] = None,
    missing_rank_policy: str = "zero",
    pool_depth: int = 50,
) -> List[Tuple[str, float, Dict[str, Optional[int]]]]:
    """
    Computes missing-rank-aware Reciprocal Rank Fusion over canonical record_ids.
    """
    if missing_rank_policy not in ("zero", "floor_k_plus_1"):
        raise ValueError(f"Unsupported missing_rank_policy: {missing_rank_policy}")

    weights = weights or {m: 1.0 for m in ranked_lists}
    candidate_union: set[str] = set()
    rank_lookup: Dict[str, Dict[str, int]] = {}

    for model_key, records in ranked_lists.items():
        deduped: List[str] = []
        seen: set[str] = set()
        for rid in records[:pool_depth]:
            if rid not in seen:
                seen.add(rid)
                deduped.append(rid)
                candidate_union.add(rid)
        rank_lookup[model_key] = {rid: idx + 1 for idx, rid in enumerate(deduped)}

    results: List[Tuple[str, float, Dict[str, Optional[int]]]] = []
    for rid in sorted(candidate_union):
        score = 0.0
        per_model_ranks: Dict[str, Optional[int]] = {}
        for model_key, rmap in rank_lookup.items():
            w = weights.get(model_key, 1.0)
            if rid in rmap:
                r = rmap[rid]
                per_model_ranks[model_key] = r
                score += w * (1.0 / (k + r))
            else:
                per_model_ranks[model_key] = None
                if missing_rank_policy == "floor_k_plus_1":
                    score += w * (1.0 / (k + pool_depth + 1))
        results.append((rid, score, per_model_ranks))

    results.sort(key=lambda item: (-item[1], item[0]))
    return results


@dataclass
class StormEventSegment:
    storm_event_id: str
    start_epoch_sec: int
    end_epoch_sec: int
    partition: str  # "TRAIN" | "VALIDATION" | "TEST"
    peak_precip_mm_hr: float
    peak_moisture_flux: float


def validate_event_isolated_splits(
    events: List[StormEventSegment],
    embargo_hours: int = 72,
) -> Tuple[bool, List[str]]:
    """
    Verifies that:
    1. Every storm_event_id belongs to a single partition (atomic event assignment).
    2. Chronological boundaries between TRAIN -> VALIDATION -> TEST maintain at least
       `embargo_hours` (72h) separation.
    """
    violations: List[str] = []
    event_partitions: Dict[str, set[str]] = {}
    for ev in events:
        event_partitions.setdefault(ev.storm_event_id, set()).add(ev.partition)

    for eid, parts in event_partitions.items():
        if len(parts) > 1:
            violations.append(f"Storm event {eid} leaked across partitions: {sorted(parts)}")

    embargo_sec = embargo_hours * 3600
    by_part: Dict[str, List[StormEventSegment]] = {"TRAIN": [], "VALIDATION": [], "TEST": []}
    for ev in events:
        if ev.partition in by_part:
            by_part[ev.partition].append(ev)

    if by_part["TRAIN"] and by_part["VALIDATION"]:
        max_train_end = max(e.end_epoch_sec for e in by_part["TRAIN"])
        min_val_start = min(e.start_epoch_sec for e in by_part["VALIDATION"])
        if min_val_start - max_train_end < embargo_sec:
            violations.append(
                f"TRAIN->VALIDATION gap {(min_val_start - max_train_end)/3600:.1f}h < {embargo_hours}h embargo"
            )

    if by_part["VALIDATION"] and by_part["TEST"]:
        max_val_end = max(e.end_epoch_sec for e in by_part["VALIDATION"])
        min_test_start = min(e.start_epoch_sec for e in by_part["TEST"])
        if min_test_start - max_val_end < embargo_sec:
            violations.append(
                f"VALIDATION->TEST gap {(min_test_start - max_val_end)/3600:.1f}h < {embargo_hours}h embargo"
            )

    return (len(violations) == 0, violations)


@dataclass
class ClaimBinding:
    source_field: str
    variable_name: str
    reported_value: float
    unit: str
    horizon_minutes: int
    location_id: str
    provenance_class: str


@dataclass
class VerifiedClaim:
    claim_id: str
    statement_prefix: str
    statement_suffix: str
    bindings: List[ClaimBinding]


def verify_structured_provenance(
    claims: List[VerifiedClaim],
    context_registry: Dict[str, Dict[str, Any]],
    qualitative_notes: List[str],
    tolerance: float = 1e-2,
) -> Tuple[bool, List[str]]:
    """
    8-Rule Deterministic Provenance Verifier for Gemma 4 structured forecast briefings.
    """
    errors: List[str] = []

    for claim in claims:
        # Check prefix/suffix for unbound floating-point literals
        for text_part in (claim.statement_prefix, claim.statement_suffix):
            unbound_floats = re.findall(r"\b\d+\.\d+\b", text_part)
            if unbound_floats:
                errors.append(
                    f"[Rule 8 Violation] Claim {claim.claim_id} contains unbound float literal(s) {unbound_floats} in narrative prose."
                )

        for b in claim.bindings:
            # Rule 1: Path existence
            if b.source_field not in context_registry:
                errors.append(f"[Rule 1 Violation] Missing source_field '{b.source_field}' in claim {claim.claim_id}.")
                continue

            truth = context_registry[b.source_field]
            # Rule 2: Variable semantic correspondence
            if b.variable_name != truth["variable_name"]:
                errors.append(
                    f"[Rule 2 Violation] Field '{b.source_field}' expects variable '{truth['variable_name']}', got '{b.variable_name}'."
                )
            # Rule 3: Unit consistency
            if b.unit != truth["unit"]:
                errors.append(
                    f"[Rule 3 Violation] Field '{b.source_field}' expects unit '{truth['unit']}', got '{b.unit}'."
                )
            # Rule 4: Forecast horizon match
            if b.horizon_minutes != truth["horizon_minutes"]:
                errors.append(
                    f"[Rule 4 Violation] Field '{b.source_field}' expects horizon {truth['horizon_minutes']}m, got {b.horizon_minutes}m."
                )
            # Rule 5: Geographic scope match
            if b.location_id != truth["location_id"]:
                errors.append(
                    f"[Rule 5 Violation] Field '{b.source_field}' expects location '{truth['location_id']}', got '{b.location_id}'."
                )
            # Rule 6: Provenance class match
            if b.provenance_class != truth["provenance_class"]:
                errors.append(
                    f"[Rule 6 Violation] Field '{b.source_field}' expects provenance '{truth['provenance_class']}', got '{b.provenance_class}'."
                )
            # Rule 7: Numeric tolerance
            if abs(b.reported_value - float(truth["value"])) > tolerance:
                errors.append(
                    f"[Rule 7 Violation] Value mismatch on '{b.source_field}': reported {b.reported_value} vs truth {truth['value']}."
                )

    for note in qualitative_notes:
        unbound_floats = re.findall(r"\b\d+\.\d+\b", note)
        if unbound_floats:
            errors.append(f"[Rule 8 Violation] Qualitative note contains unbound float literal(s): {unbound_floats}")

    return (len(errors) == 0, errors)
