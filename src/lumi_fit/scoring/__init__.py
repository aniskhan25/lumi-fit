"""Explainable scoring and deterministic candidate ranking."""

from .engine import enrich_candidates, rank_candidates, serving_targets_satisfied

__all__ = ["enrich_candidates", "rank_candidates", "serving_targets_satisfied"]
