"""Store, consent, provenance and the Measured / Estimated / Reported separation."""
import re
import sqlite3
from pathlib import Path

import pytest

from src.neurotwin.store import ConsentError, Store

SRC = Path(__file__).resolve().parents[1] / "src" / "neurotwin"


@pytest.fixture
def store():
    s = Store(":memory:")
    s.ensure_user("u")
    return s


def test_nothing_is_stored_without_consent(store):
    for kind in ("mood", "reason", "free_text"):
        with pytest.raises(ConsentError):
            store.add_reported("u", kind, "x")
    assert store.list_reported("u") == []


def test_each_kind_needs_its_own_consent(store):
    store.set_consent("u", store_checkins=True)
    store.add_reported("u", "mood", "tense")
    with pytest.raises(ConsentError):
        store.add_reported("u", "reason", "studies")        # needs store_stressors
    with pytest.raises(ConsentError):
        store.add_reported("u", "free_text", "hello")       # needs store_free_text
    store.set_consent("u", store_stressors=True, store_free_text=True)
    store.add_reported("u", "reason", "studies")
    store.add_reported("u", "free_text", "hello")


def test_database_refuses_model_output_written_as_user_report(store):
    with pytest.raises(sqlite3.IntegrityError):
        store.conn.execute(
            "INSERT INTO reported(user_id,ts,kind,value_json,provenance) VALUES('u','t','mood','1','model')")


def test_database_refuses_user_input_written_as_sensor_or_model(store):
    with pytest.raises(sqlite3.IntegrityError):
        store.conn.execute("INSERT INTO measured(user_id,ts,features_json,provenance) VALUES('u','t','{}','user')")
    with pytest.raises(sqlite3.IntegrityError):
        store.conn.execute(
            "INSERT INTO estimates(user_id,ts,model_name,result_json,provenance) VALUES('u','t','m','{}','user')")


def test_unknown_reported_kind_rejected(store):
    store.set_consent("u", store_checkins=True)
    with pytest.raises(ValueError):
        store.add_reported("u", "diagnosis", "x")


def test_layers_have_separate_tables(store):
    tables = {r[0] for r in store.conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    assert {"measured", "estimates", "reported", "recommendations", "feedback"} <= tables


def test_only_store_module_writes_to_layer_tables():
    """Static guarantee: no other module can insert into a layer table directly."""
    for path in SRC.glob("*.py"):
        text = path.read_text(encoding="utf-8")
        for table in ("reported", "estimates", "measured"):
            hits = len(re.findall(rf"INSERT INTO {table}\b", text))
            if path.name == "store.py":
                assert hits == 1, (path.name, table, hits)
            else:
                assert hits == 0, (path.name, table, hits)


def test_estimator_recommender_dialogue_do_not_import_store():
    for name in ("estimator.py", "recommender.py", "dialogue.py", "patterns.py", "simulation.py"):
        text = (SRC / name).read_text(encoding="utf-8")
        assert not re.search(r"^\s*(from|import)\s+.*\bstore\b", text, re.M), name


def test_user_can_inspect_correct_and_delete(store):
    store.set_consent("u", store_checkins=True, store_stressors=True)
    rid = store.add_reported("u", "reason", "work")
    store.correct_reported("u", rid, "studies")
    row = store.list_reported("u")[0]
    assert row["value"] == "studies" and row["edited_at"]
    assert store.export_user("u")["reported"][0]["value"] == "studies"
    assert store.delete_reported("u", rid) is True
    assert store.list_reported("u") == []


def test_users_cannot_touch_each_others_reports(store):
    store.ensure_user("v")
    store.set_consent("u", store_checkins=True)
    rid = store.add_reported("u", "mood", "tense")
    with pytest.raises(KeyError):
        store.correct_reported("v", rid, "calm_ok")
    assert store.delete_reported("v", rid) is False


def test_delete_user_removes_everything(store):
    store.set_consent("u", store_checkins=True, store_feedback=True, store_stressors=True)
    store.add_reported("u", "mood", "tense")
    rec = store.add_recommendation("u", "who_grounding", 1, ["r"])
    store.add_feedback("u", rec, "yes", 4)
    store.set_preference("u", "item:who_grounding", 1.0)
    store.delete_user("u")
    assert store.conn.execute("SELECT COUNT(*) FROM reported").fetchone()[0] == 0
    assert store.conn.execute("SELECT COUNT(*) FROM feedback").fetchone()[0] == 0
    assert store.conn.execute("SELECT COUNT(*) FROM recommendations").fetchone()[0] == 0
    assert store.conn.execute("SELECT COUNT(*) FROM preferences").fetchone()[0] == 0


def test_feedback_requires_consent_and_ownership(store):
    store.set_consent("u", store_checkins=True)
    rec = store.add_recommendation("u", "who_grounding", 1, [])
    with pytest.raises(ConsentError):
        store.add_feedback("u", rec, "yes")
    store.set_consent("u", store_feedback=True)
    store.ensure_user("v")
    store.set_consent("v", store_feedback=True)
    with pytest.raises(KeyError):
        store.add_feedback("v", rec, "yes")                 # someone else's recommendation
    store.add_feedback("u", rec, "yes", 3)


def test_feedback_value_ranges_enforced(store):
    store.set_consent("u", store_checkins=True, store_feedback=True)
    rec = store.add_recommendation("u", "who_grounding", 1, [])
    with pytest.raises(sqlite3.IntegrityError):
        store.add_feedback("u", rec, "yes", feel_after=11)
    with pytest.raises(sqlite3.IntegrityError):
        store.add_feedback("u", rec, "maybe", 3)


def test_session_stores_are_independent_and_not_on_disk(tmp_path):
    a, b = Store(":memory:"), Store(":memory:")
    a.ensure_user("u"); a.set_consent("u", store_checkins=True)
    a.add_reported("u", "mood", "tense")
    b.ensure_user("u")
    assert b.list_reported("u") == []
    assert not list(tmp_path.iterdir())


def test_reset_preferences(store):
    store.set_consent("u", store_feedback=True)
    store.set_preference("u", "item:x", 1.0)
    store.reset_preferences("u")
    assert store.get_preferences("u") == {}
