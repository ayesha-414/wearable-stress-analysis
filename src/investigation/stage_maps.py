"""
src/investigation/stage_maps.py
===============================
Notebook-consistent stage definitions for V1 and V2 cohorts.

Citation and provenance:
- Wearable_Dataset.ipynb Cell 25:
    In the authors' notebook, visualization highlights task windows using:
    V1 ('S' cohort):
        Stroop: tags[3] to tags[4]
        TMCT: tags[5] to tags[6]
        Real Opinion: tags[7] to tags[8]
        Opposite Opinion: tags[9] to tags[10]
        Subtract: tags[11] to tags[12]
    V2 ('f' cohort):
        TMCT: tags[2] to tags[3]
        Real Opinion: tags[4] to tags[5]
        Opposite Opinion: tags[6] to tags[7]
        Subtract: tags[8] to tags[9]
- Wearable_Dataset.ipynb Cell 27:
    The notebook inserts `utc_start_dict[folder_name]` (the EDA start timestamp)
    at index 0 of `tags`, so notebook tag indices are shifted by +1 relative to raw tags.
    Therefore:
    raw_tag_index = notebook_tag_index - 1.
- Stage audit verification:
    Verified empirically in `data/processed/model_investigation/stage_audit/stage_audit_comparison_table.csv`
    and `alignment_diagnostic_*.txt` Section 5.
    Task durations in the notebook-consistent map match the clinical protocol durations:
    - V1 Stroop: median 109 s
    - V1 TMCT: median 137 s
    - V1 Opinion / Subtract tasks: median 31-32 s
    - V2 TMCT: median 261 s
    - V2 Opinion / Subtract tasks: median 38 s
"""

from __future__ import annotations

# Notebook-consistent V1 stages mapped to raw tag indices
# Follows chronological order of Stress_Level_v1.csv:
# ['Baseline', 'Stroop', 'First Rest', 'TMCT', 'Second Rest', 'Real Opinion', 'Opposite Opinion', 'Subtract']
STRESS_V1_STAGES_V2 = {
    "Baseline":         (0, 2),   # Notebook [1, 3] -> raw (0, 2); median ~287 s
    "Stroop":           (2, 3),   # Notebook [3, 4] -> raw (2, 3); median 109 s (task)
    "First Rest":       (3, 4),   # Post-Stroop rest -> raw (3, 4); median 326 s (~5 min rest)
    "TMCT":             (4, 5),   # Notebook [5, 6] -> raw (4, 5); median 137 s (task)
    "Second Rest":      (5, 6),   # Post-TMCT rest -> raw (5, 6); median 308 s (~5 min rest)
    "Real Opinion":     (6, 7),   # Notebook [7, 8] -> raw (6, 7); median 32 s (task)
    "Opposite Opinion": (8, 9),   # Notebook [9, 10] -> raw (8, 9); median 31 s (task)
    "Subtract":         (10, 11), # Notebook [11, 12] -> raw (10, 11); median 32 s (task)
}

# Notebook-consistent V2 stages mapped to raw tag indices
# Follows chronological order of Stress_Level_v2.csv:
# ['Baseline', 'TMCT', 'First Rest', 'Real Opinion', 'Opposite Opinion', 'Second Rest', 'Subtract']
STRESS_V2_STAGES_V2 = {
    "Baseline":         (0, 1),   # Notebook [1, 2] -> raw (0, 1); median 98 s
    "TMCT":             (1, 2),   # Notebook [2, 3] -> raw (1, 2); median 261 s (task)
    "First Rest":       (2, 3),   # Post-TMCT rest -> raw (2, 3); median 853 s (~14 min rest)
    "Real Opinion":     (3, 4),   # Notebook [4, 5] -> raw (3, 4); median 38 s (task)
    "Opposite Opinion": (5, 6),   # Notebook [6, 7] -> raw (5, 6); median 38 s (task)
    "Second Rest":      (6, 7),   # Post-Opinion rest -> raw (6, 7); median 800 s (~13 min rest)
    "Subtract":         (7, 8),   # Notebook [8, 9] -> raw (7, 8); median 38 s (task)
}

# Expected duration ranges for sanity checking (min_s, max_s)
EXPECTED_DURATIONS_V1 = {
    "Baseline": (150.0, 450.0),
    "Stroop": (80.0, 180.0),
    "First Rest": (200.0, 450.0),
    "TMCT": (100.0, 200.0),
    "Second Rest": (200.0, 450.0),
    "Real Opinion": (20.0, 60.0),
    "Opposite Opinion": (20.0, 60.0),
    "Subtract": (20.0, 60.0),
}

EXPECTED_DURATIONS_V2 = {
    "Baseline": (60.0, 200.0),
    "TMCT": (180.0, 480.0),
    "First Rest": (500.0, 1200.0),
    "Real Opinion": (20.0, 60.0),
    "Opposite Opinion": (20.0, 60.0),
    "Second Rest": (500.0, 1200.0),
    "Subtract": (20.0, 70.0),
}


def get_stage_map_v2(activity: str, subject_id: str) -> dict[str, tuple[int, int]]:
    """Return the notebook-consistent stage map for subject and activity."""
    if activity != "STRESS":
        raise ValueError(f"v2 stage map only defined for STRESS activity, got {activity}")
    if subject_id.startswith("f"):
        return STRESS_V2_STAGES_V2
    return STRESS_V1_STAGES_V2

