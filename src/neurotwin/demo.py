"""Adapter that presents windows of the RESEARCH dataset as a demo 'participant'.

These are recordings from healthy volunteers in a laboratory stress protocol. They are NOT a
live user and not the person using the app. The 1-10 rating stored in the dataset was
REPORTED by the volunteer during the protocol and is shown as such (layer: reported, setting:
laboratory), alongside the model's out-of-sample estimate (layer: estimated).
"""
from __future__ import annotations

import pandas as pd

from src.models import FEATURE_COLS, STRESS_THRESHOLD

from .estimator import StressEstimator

DEMO_NOTICE = ("Demo data: these windows come from healthy volunteers in a laboratory stress protocol "
               "(PhysioNet Wearable Device Dataset). They are not a live recording of you.")


class DemoData:
    def __init__(self, estimator: StressEstimator):
        self.est = estimator
        self.df = estimator.raw
        self.index = estimator.X.index

    def participants(self) -> list[str]:
        return sorted(self.est.groups.unique())

    def windows(self, subject: str) -> pd.DataFrame:
        idx = [i for i in self.index if self.est.groups.loc[i] == subject]
        return self.df.loc[idx, [c for c in ("subject", "stage") if c in self.df.columns]
                           + ["stress_level"]].copy()

    def measured(self, index) -> dict:
        row = self.est.X.loc[index]
        return {c: (None if pd.isna(row[c]) else float(row[c])) for c in row.index}

    def lab_report(self, index) -> dict:
        """What the volunteer reported at that stage in the laboratory (rating 1-10)."""
        v = self.df.loc[index, "stress_level"]
        return {"rating": None if pd.isna(v) else float(v),
                "at_or_above_threshold": None if pd.isna(v) else bool(v >= STRESS_THRESHOLD),
                "setting": "laboratory protocol, verbal 1-10 rating"}

    def baseline(self, subject: str) -> dict:
        return self.est.baseline_summary(self.df, subject)
