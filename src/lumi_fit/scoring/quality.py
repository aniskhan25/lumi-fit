"""Task-quality evidence kept separate from LUMI serving fit."""

from __future__ import annotations

from decimal import Decimal

from lumi_fit.domain.models import ModelRecord
from lumi_fit.domain.requests import UseCase
from lumi_fit.domain.results import ConfidenceClass, ModelSuitabilityEvidence


def model_suitability(model: ModelRecord, use_case: UseCase) -> ModelSuitabilityEvidence:
    """Normalize cataloged quality evidence for one task without mixing it into fit."""

    matching = [item for item in model.evaluations if item.use_case == use_case.value]
    if not matching:
        return ModelSuitabilityEvidence(
            use_case=use_case,
            score=None,
            confidence=ConfidenceClass.UNKNOWN,
            explanation=(
                "Capability metadata matches, but no normalized task-quality "
                "evaluation is cataloged."
            ),
        )
    best = max(
        matching,
        key=lambda item: (
            (item.score - item.score_minimum) / (item.score_maximum - item.score_minimum)
        ),
    )
    normalized = (
        Decimal(100) * (best.score - best.score_minimum) / (best.score_maximum - best.score_minimum)
    )
    return ModelSuitabilityEvidence(
        use_case=use_case,
        score=normalized,
        confidence=ConfidenceClass.MEASURED_NEAR,
        evidence_ids=(best.evaluation_id,),
        explanation="Normalized from the cataloged task-quality evaluation range.",
    )
