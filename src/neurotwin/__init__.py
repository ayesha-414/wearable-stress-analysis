"""NeuroTwin support layer.

Adds, on top of the existing research code (``src/features.py``,
``src/models.py``), the parts of the project that go beyond stress classification:

* ``estimator``   - explainable, uncertainty-aware wrapper around the EXISTING models
* ``dialogue``    - scripted, consent-aware conversational check-in
* ``safety``      - crisis screening, region-configurable resources, prompt policy
* ``store``       - local SQLite profile that keeps Measured / Estimated / Reported apart
* ``catalogue`` / ``recommender`` - sourced coping catalogue and a transparent rule-based recommender
* ``patterns``    - cautious pattern reporting from the person's OWN reports
* ``profile``     - the "digital twin" profile view-model
* ``simulation``  - scripted simulated users (software verification ONLY)

Nothing here modifies feature formulas, model definitions or saved results.
This is a research prototype, not a medical device.
"""

__version__ = "0.1.0"

DISCLAIMER = (
    "NeuroTwin AI is a research prototype. It is not a medical device, does not "
    "diagnose any condition, and does not replace a mental health professional."
)
