"""Headless end-to-end tests of the Streamlit app (all pages + the key conversation paths)."""
import logging
import warnings
from pathlib import Path

import pytest

pytest.importorskip("streamlit")
pytest.importorskip("plotly")
from streamlit.testing.v1 import AppTest  # noqa: E402

from src.neurotwin.ui import SUPPORT_PAGES  # noqa: E402

warnings.filterwarnings("ignore")
logging.disable(logging.CRITICAL)
APP = str(Path(__file__).resolve().parents[1] / "dashboard" / "app.py")
RESEARCH_PAGES = ["Overview", "Analyzer", "Signal Lab", "Explainable AI", "Model Arena",
                  "Personal Twin", "Methods"]


def fresh() -> AppTest:
    at = AppTest.from_file(APP, default_timeout=180).run()
    assert not at.exception, [e.value for e in at.exception]
    return at


def radio(at, label):
    return [r for r in at.sidebar.radio if r.label == label][0]


def click(at, label):
    b = [x for x in at.button if x.label == label]
    assert b, f"no button {label!r}; have {[x.label for x in at.button]}"
    b[0].click().run()
    assert not at.exception, [e.value for e in at.exception]


def test_all_ten_support_pages_render():
    assert len(SUPPORT_PAGES) == 10
    at = fresh()
    for p in SUPPORT_PAGES:
        radio(at, "Navigation").set_value(p).run()
        assert not at.exception, (p, [e.value for e in at.exception])


def test_existing_research_pages_still_render():
    at = fresh()
    radio(at, "Section").set_value("Research tools").run()
    for p in RESEARCH_PAGES:
        radio(at, "Navigation").set_value(p).run()
        assert not at.exception, (p, [e.value for e in at.exception])


def test_urgent_help_is_available_in_the_sidebar_on_every_page():
    at = fresh()
    for p in SUPPORT_PAGES:
        radio(at, "Navigation").set_value(p).run()
        labels = [e.label for e in at.sidebar.expander]
        assert any("Talk to someone now" in l for l in labels), p


def test_home_shows_three_layers_and_no_alarming_wording():
    at = fresh()
    text = " ".join(m.value for m in at.markdown)
    for layer in ("Measured", "Estimated", "Reported"):
        assert layer in text
    assert "you are stressed" not in text.lower()


def test_full_conversation_stores_user_reports_and_feedback_with_correct_provenance():
    at = fresh()
    radio(at, "Navigation").set_value("Check-in").run()
    click(at, "Start a check-in")
    click(at, "Overwhelmed")
    click(at, "Continue")                          # rating slider
    click(at, "A practical suggestion")
    click(at, "Studies")
    click(at, "Yes")                               # felt this before
    s = at.session_state["nt_session"]
    assert s.node_id == "recommend"
    shown = at.session_state["nt_shown"]
    assert shown and len(shown) <= 3
    at.selectbox(key=f"pick-{s.id}").set_value(shown[0].item.title).run()
    click(at, "Continue")
    click(at, "Yes")                               # helped
    click(at, "Continue")                          # how do you feel now
    click(at, "No")                                # different?
    click(at, "Yes")                               # again in future?
    store = at.session_state["nt_mem_store"]
    reported = store.list_reported("local_user")
    assert {r["kind"] for r in reported} >= {"mood", "stress_rating", "wants", "reason"}
    assert all(r["provenance"] == "user" for r in reported)
    assert all(r["provenance"] == "system" for r in store.list_recommendations("local_user"))
    assert store.list_feedback("local_user")[0]["helped"] == "yes"
    assert store.get_preferences("local_user").get(f"item:{shown[0].item.id}", 0) > 0
    assert store.list_estimates("local_user") == []        # demo windows are never added to a profile


def test_urgent_help_stops_everything_and_shows_support():
    at = fresh()
    radio(at, "Navigation").set_value("Check-in").run()
    click(at, "Start a check-in")
    click(at, "Tense or uneasy")
    click(at, "🆘 I need urgent help")
    s = at.session_state["nt_session"]
    assert s.crisis and ("nt_shown" not in at.session_state or at.session_state["nt_shown"] is None)
    assert any("real person" in e.value for e in at.error)
    store = at.session_state["nt_mem_store"]
    assert store.list_recommendations("local_user") == []


def test_user_can_end_the_checkin_at_any_time_and_nothing_more_is_asked():
    at = fresh()
    radio(at, "Navigation").set_value("Check-in").run()
    click(at, "Start a check-in")
    click(at, "End check-in")
    assert at.session_state["nt_session"].declined
    store = at.session_state["nt_mem_store"]
    assert store.list_reported("local_user") == []


def test_nothing_is_written_to_disk_by_default(tmp_path):
    from src.neurotwin import config
    before = config.PROFILE_DB.exists()
    at = fresh()
    radio(at, "Navigation").set_value("Check-in").run()
    click(at, "Start a check-in")
    click(at, "Calm or okay")
    assert config.PROFILE_DB.exists() == before


def test_evaluation_page_shows_baseline_and_limitations():
    at = fresh()
    radio(at, "Navigation").set_value("Evaluation & Transparency").run()
    text = " ".join([m.value for m in at.markdown] + [w.value for w in at.warning])
    assert "baseline" in text.lower() and "does not claim" in text.lower()
    assert any("majority" in m.label.lower() for m in at.metric)
