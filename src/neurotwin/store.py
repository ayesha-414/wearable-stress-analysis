"""Local SQLite profile store.

Design rules
------------
1. Measured, Estimated and Reported data live in SEPARATE tables, and each table
   has a CHECK constraint on ``provenance`` (sensor / model / user / system). The
   database itself therefore refuses, for example, a model output written as a
   user report. There is no generic "insert anything" method.
2. Writing to the Reported layer requires consent (see ``KIND_CONSENT``). Without
   consent a ``ConsentError`` is raised and nothing is written.
3. The user can inspect (``export_user``), correct (``correct_reported``) and
   delete (``delete_reported``, ``delete_user``) their own data.
4. Use ``Store(":memory:")`` for session-only data that disappears when the
   session ends. The on-disk database is created only when a path is given.

This is a prototype store: no encryption at rest, no authentication. It is not
suitable for real deployment without a data-protection review.
"""
from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

REPORTED_KINDS = (
    "mood", "stress_rating", "reason", "similar_before", "estimate_agreement",
    "free_text", "sleep_hours", "wants", "context_note",
)
# Which consent flag each reported kind needs.
KIND_CONSENT = {
    "mood": "store_checkins", "stress_rating": "store_checkins",
    "similar_before": "store_checkins", "estimate_agreement": "store_checkins",
    "sleep_hours": "store_checkins", "wants": "store_checkins",
    "context_note": "store_checkins",
    "reason": "store_stressors",
    "free_text": "store_free_text",
}
CONSENT_FLAGS = ("store_checkins", "store_stressors", "store_free_text",
                 "store_feedback", "prompts_enabled")


class ConsentError(PermissionError):
    """Raised when data would be stored without the user's consent."""


_SCHEMA = f"""
CREATE TABLE IF NOT EXISTS users(
  user_id TEXT PRIMARY KEY, created_at TEXT NOT NULL, region TEXT);
CREATE TABLE IF NOT EXISTS consent(
  user_id TEXT PRIMARY KEY,
  store_checkins INTEGER NOT NULL DEFAULT 0,
  store_stressors INTEGER NOT NULL DEFAULT 0,
  store_free_text INTEGER NOT NULL DEFAULT 0,
  store_feedback INTEGER NOT NULL DEFAULT 0,
  prompts_enabled INTEGER NOT NULL DEFAULT 0,
  updated_at TEXT);
CREATE TABLE IF NOT EXISTS measured(
  id INTEGER PRIMARY KEY AUTOINCREMENT, user_id TEXT NOT NULL, ts TEXT NOT NULL,
  window_label TEXT, features_json TEXT NOT NULL, quality_json TEXT, source_note TEXT,
  provenance TEXT NOT NULL CHECK(provenance = 'sensor'));
CREATE TABLE IF NOT EXISTS estimates(
  id INTEGER PRIMARY KEY AUTOINCREMENT, user_id TEXT NOT NULL, ts TEXT NOT NULL,
  measured_id INTEGER, model_name TEXT NOT NULL, result_json TEXT NOT NULL,
  provenance TEXT NOT NULL CHECK(provenance = 'model'));
CREATE TABLE IF NOT EXISTS reported(
  id INTEGER PRIMARY KEY AUTOINCREMENT, user_id TEXT NOT NULL, ts TEXT NOT NULL,
  kind TEXT NOT NULL CHECK(kind IN ({",".join("'%s'" % k for k in REPORTED_KINDS)})),
  value_json TEXT NOT NULL, session_id TEXT, edited_at TEXT,
  provenance TEXT NOT NULL CHECK(provenance = 'user'));
CREATE TABLE IF NOT EXISTS recommendations(
  id INTEGER PRIMARY KEY AUTOINCREMENT, user_id TEXT NOT NULL, ts TEXT NOT NULL,
  item_id TEXT NOT NULL, rank INTEGER, reasons_json TEXT, session_id TEXT,
  provenance TEXT NOT NULL CHECK(provenance = 'system'));
CREATE TABLE IF NOT EXISTS feedback(
  id INTEGER PRIMARY KEY AUTOINCREMENT, user_id TEXT NOT NULL, rec_id INTEGER NOT NULL,
  ts TEXT NOT NULL,
  helped TEXT CHECK(helped IN ('yes','somewhat','no','skipped')),
  feel_after INTEGER CHECK(feel_after IS NULL OR (feel_after BETWEEN 0 AND 10)),
  want_different INTEGER, want_future INTEGER,
  provenance TEXT NOT NULL CHECK(provenance = 'user'));
CREATE TABLE IF NOT EXISTS preferences(
  user_id TEXT NOT NULL, key TEXT NOT NULL, score REAL NOT NULL, updated_at TEXT,
  PRIMARY KEY(user_id, key));
"""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class Store:
    def __init__(self, path: str | Path = ":memory:"):
        self.path = str(path)
        if self.path != ":memory:":
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(self.path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript(_SCHEMA)
        self.conn.commit()

    def close(self) -> None:
        self.conn.close()

    # ---------------------------------------------------------------- users / consent
    def ensure_user(self, user_id: str, region: str | None = None) -> None:
        self.conn.execute(
            "INSERT OR IGNORE INTO users(user_id, created_at, region) VALUES(?,?,?)",
            (user_id, _now(), region))
        self.conn.execute("INSERT OR IGNORE INTO consent(user_id, updated_at) VALUES(?,?)",
                          (user_id, _now()))
        if region is not None:
            self.conn.execute("UPDATE users SET region=? WHERE user_id=?", (region, user_id))
        self.conn.commit()

    def get_region(self, user_id: str) -> str | None:
        row = self.conn.execute("SELECT region FROM users WHERE user_id=?", (user_id,)).fetchone()
        return row["region"] if row else None

    def get_consent(self, user_id: str) -> dict:
        self.ensure_user(user_id)
        row = self.conn.execute("SELECT * FROM consent WHERE user_id=?", (user_id,)).fetchone()
        return {k: bool(row[k]) for k in CONSENT_FLAGS}

    def set_consent(self, user_id: str, **flags: bool) -> dict:
        self.ensure_user(user_id)
        for k, v in flags.items():
            if k not in CONSENT_FLAGS:
                raise ValueError(f"Unknown consent flag {k!r}")
            self.conn.execute(f"UPDATE consent SET {k}=?, updated_at=? WHERE user_id=?",
                              (int(bool(v)), _now(), user_id))
        self.conn.commit()
        return self.get_consent(user_id)

    def _require(self, user_id: str, flag: str) -> None:
        if not self.get_consent(user_id).get(flag, False):
            raise ConsentError(f"Consent '{flag}' is off; nothing was stored.")

    # ---------------------------------------------------------------- MEASURED (sensor)
    def add_measured(self, user_id: str, features: dict, quality: dict | None = None,
                     window_label: str | None = None, source_note: str | None = None) -> int:
        self._require(user_id, "store_checkins")
        cur = self.conn.execute(
            "INSERT INTO measured(user_id,ts,window_label,features_json,quality_json,source_note,provenance)"
            " VALUES(?,?,?,?,?,?,'sensor')",
            (user_id, _now(), window_label, json.dumps(features, default=_jd),
             json.dumps(quality or {}, default=_jd), source_note))
        self.conn.commit()
        return cur.lastrowid

    # ---------------------------------------------------------------- ESTIMATED (model)
    def add_estimate(self, user_id: str, model_name: str, result: dict,
                     measured_id: int | None = None) -> int:
        self._require(user_id, "store_checkins")
        cur = self.conn.execute(
            "INSERT INTO estimates(user_id,ts,measured_id,model_name,result_json,provenance)"
            " VALUES(?,?,?,?,?,'model')",
            (user_id, _now(), measured_id, model_name, json.dumps(result, default=_jd)))
        self.conn.commit()
        return cur.lastrowid

    # ---------------------------------------------------------------- REPORTED (user)
    def add_reported(self, user_id: str, kind: str, value, session_id: str | None = None,
                     ts: str | None = None) -> int:
        if kind not in REPORTED_KINDS:
            raise ValueError(f"Unknown reported kind {kind!r}")
        self._require(user_id, KIND_CONSENT[kind])
        cur = self.conn.execute(
            "INSERT INTO reported(user_id,ts,kind,value_json,session_id,provenance)"
            " VALUES(?,?,?,?,?,'user')",
            (user_id, ts or _now(), kind, json.dumps(value, default=_jd), session_id))
        self.conn.commit()
        return cur.lastrowid

    def list_reported(self, user_id: str, kinds: tuple[str, ...] | None = None) -> list[dict]:
        q = "SELECT * FROM reported WHERE user_id=?"
        args: list = [user_id]
        if kinds:
            q += f" AND kind IN ({','.join('?' * len(kinds))})"
            args += list(kinds)
        q += " ORDER BY ts, id"
        return [self._row(r, "value_json") for r in self.conn.execute(q, args)]

    def correct_reported(self, user_id: str, reported_id: int, new_value) -> None:
        cur = self.conn.execute(
            "UPDATE reported SET value_json=?, edited_at=? WHERE id=? AND user_id=?",
            (json.dumps(new_value, default=_jd), _now(), reported_id, user_id))
        if cur.rowcount == 0:
            raise KeyError("No such report for this user.")
        self.conn.commit()

    def delete_reported(self, user_id: str, reported_id: int) -> bool:
        cur = self.conn.execute("DELETE FROM reported WHERE id=? AND user_id=?",
                                (reported_id, user_id))
        self.conn.commit()
        return cur.rowcount > 0

    # ---------------------------------------------------------------- recommendations / feedback
    def add_recommendation(self, user_id: str, item_id: str, rank: int,
                           reasons: list[str], session_id: str | None = None) -> int:
        self._require(user_id, "store_checkins")
        cur = self.conn.execute(
            "INSERT INTO recommendations(user_id,ts,item_id,rank,reasons_json,session_id,provenance)"
            " VALUES(?,?,?,?,?,?,'system')",
            (user_id, _now(), item_id, rank, json.dumps(reasons), session_id))
        self.conn.commit()
        return cur.lastrowid

    def list_recommendations(self, user_id: str) -> list[dict]:
        return [self._row(r, "reasons_json") for r in self.conn.execute(
            "SELECT * FROM recommendations WHERE user_id=? ORDER BY ts, id", (user_id,))]

    def add_feedback(self, user_id: str, rec_id: int, helped: str | None = None,
                     feel_after: int | None = None, want_different: bool | None = None,
                     want_future: bool | None = None) -> int:
        self._require(user_id, "store_feedback")
        owner = self.conn.execute("SELECT user_id FROM recommendations WHERE id=?",
                                  (rec_id,)).fetchone()
        if owner is None or owner["user_id"] != user_id:
            raise KeyError("Recommendation not found for this user.")
        cur = self.conn.execute(
            "INSERT INTO feedback(user_id,rec_id,ts,helped,feel_after,want_different,want_future,provenance)"
            " VALUES(?,?,?,?,?,?,?,'user')",
            (user_id, rec_id, _now(), helped, feel_after,
             None if want_different is None else int(want_different),
             None if want_future is None else int(want_future)))
        self.conn.commit()
        return cur.lastrowid

    def list_feedback(self, user_id: str) -> list[dict]:
        rows = self.conn.execute(
            "SELECT f.*, r.item_id FROM feedback f JOIN recommendations r ON r.id=f.rec_id "
            "WHERE f.user_id=? ORDER BY f.ts, f.id", (user_id,))
        return [dict(r) for r in rows]

    # ---------------------------------------------------------------- preferences
    def get_preferences(self, user_id: str) -> dict[str, float]:
        return {r["key"]: r["score"] for r in self.conn.execute(
            "SELECT key, score FROM preferences WHERE user_id=?", (user_id,))}

    def set_preference(self, user_id: str, key: str, score: float) -> None:
        self._require(user_id, "store_feedback")
        self.conn.execute(
            "INSERT INTO preferences(user_id,key,score,updated_at) VALUES(?,?,?,?) "
            "ON CONFLICT(user_id,key) DO UPDATE SET score=excluded.score, updated_at=excluded.updated_at",
            (user_id, key, float(score), _now()))
        self.conn.commit()

    def reset_preferences(self, user_id: str) -> None:
        """Forget everything the system has learned about this person's preferences."""
        self.conn.execute("DELETE FROM preferences WHERE user_id=?", (user_id,))
        self.conn.commit()

    # ---------------------------------------------------------------- measured/estimate reads
    def list_measured(self, user_id: str) -> list[dict]:
        return [self._row(r, "features_json", "quality_json") for r in self.conn.execute(
            "SELECT * FROM measured WHERE user_id=? ORDER BY ts, id", (user_id,))]

    def list_estimates(self, user_id: str) -> list[dict]:
        return [self._row(r, "result_json") for r in self.conn.execute(
            "SELECT * FROM estimates WHERE user_id=? ORDER BY ts, id", (user_id,))]

    # ---------------------------------------------------------------- inspect / delete
    def export_user(self, user_id: str) -> dict:
        """Everything held about the user, separated by layer (for inspection/export)."""
        return {
            "user_id": user_id,
            "region": self.get_region(user_id),
            "consent": self.get_consent(user_id),
            "measured": self.list_measured(user_id),
            "estimated": self.list_estimates(user_id),
            "reported": self.list_reported(user_id),
            "recommendations": self.list_recommendations(user_id),
            "feedback": self.list_feedback(user_id),
            "preferences": self.get_preferences(user_id),
        }

    def delete_user(self, user_id: str) -> None:
        for t in ("feedback", "recommendations", "reported", "estimates", "measured",
                  "preferences", "consent", "users"):
            self.conn.execute(f"DELETE FROM {t} WHERE user_id=?", (user_id,))
        self.conn.commit()

    # ---------------------------------------------------------------- helpers
    @staticmethod
    def _row(r: sqlite3.Row, *json_cols: str) -> dict:
        d = dict(r)
        for c in json_cols:
            if d.get(c) is not None:
                d[c.replace("_json", "")] = json.loads(d.pop(c))
        return d


def _jd(o):
    """JSON default: numpy scalars, NaN-safe."""
    try:
        import numpy as np
        if isinstance(o, np.generic):
            return o.item()
    except Exception:
        pass
    return str(o)
