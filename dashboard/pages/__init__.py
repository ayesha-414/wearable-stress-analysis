"""NeuroTwin Dashboard page modules — Aurora Edition."""

from .p01_home import render_page_home
from .p02_upload import render_page_upload
from .p02_signals import render_page_signals
from .p03_questions import render_page_questions
from .p04_result import render_page_result
from .p04_digital_twin import render_page_digital_twin
from .p05_how_calculated import render_page_how_calculated
from .p05_interventions import render_page_interventions
from .p06_model_performance import render_page_model_performance
from .p06_evaluation import render_page_evaluation
from .p07_feature_lab import render_page_feature_lab
from .p07_about import render_page_about
from .p08_qol_scale import render_page_qol_scale
from .p09_paper import render_page_research_paper

__all__ = [
    "render_page_home",
    "render_page_upload",
    "render_page_signals",
    "render_page_questions",
    "render_page_result",
    "render_page_digital_twin",
    "render_page_how_calculated",
    "render_page_interventions",
    "render_page_model_performance",
    "render_page_evaluation",
    "render_page_feature_lab",
    "render_page_about",
    "render_page_qol_scale",
    "render_page_research_paper",
]
