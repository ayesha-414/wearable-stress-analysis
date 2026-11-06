"""Page 4: Digital Twin View — NeuroTwin Aurora.

Exposes the digital twin interface and aliases render_page_result.
"""

from dashboard.pages.p04_result import render_page_result
from src.neurotwin.estimator import StressEstimator


def render_page_digital_twin(estimator: StressEstimator | None = None):
    """Render the Digital Twin interface."""
    render_page_result(estimator=estimator)

