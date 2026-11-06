"""NeuroTwin Research Extensions.

Additive research-grade modules:
- formulas: Physiological feature definitions, mathematical formulas, physiological justifications, references.
- scales: PSS-10 scoring, WHO-5 Well-Being Index, one-factor CFA model, SEM architecture, and SVM agreement meter.
- report: Publication-grade PDF report generator.
"""

from .formulas import PHYSIOLOGICAL_FEATURES, FeatureDefinition
from .report import generate_research_pdf_report
from .scales import (
    PSS_ITEMS,
    PSS_OPTIONS,
    WHO5_ITEMS,
    WHO5_OPTIONS,
    PSS10Result,
    QoLResult,
    assess_agreement,
    calculate_one_factor_cfa,
    compute_svm_model_score,
    get_sem_path_model,
    save_questionnaire_to_store,
    score_pss10,
    score_qol,
)

__all__ = [
    "PHYSIOLOGICAL_FEATURES",
    "FeatureDefinition",
    "PSS_ITEMS",
    "PSS_OPTIONS",
    "WHO5_ITEMS",
    "WHO5_OPTIONS",
    "PSS10Result",
    "QoLResult",
    "score_pss10",
    "score_qol",
    "calculate_one_factor_cfa",
    "get_sem_path_model",
    "compute_svm_model_score",
    "assess_agreement",
    "save_questionnaire_to_store",
    "generate_research_pdf_report",
]

