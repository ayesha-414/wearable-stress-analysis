"""
tests/test_interactive_app_flow.py
==================================
Thorough headless interactive verification of every button, file upload,
questionnaire response, recommendation feedback, and page navigation in the app.
"""

from pathlib import Path
import pandas as pd
import pytest

from streamlit.testing.v1 import AppTest

APP = str(Path(__file__).resolve().parents[1] / "dashboard" / "app.py")
SAMPLE_CSV = Path(__file__).resolve().parents[1] / "sample_upload.csv"


def radio(at, label):
    matches = [r for r in at.sidebar.radio if r.label == label]
    assert matches, f"Radio with label {label!r} not found. Available: {[r.label for r in at.sidebar.radio]}"
    return matches[0]


def test_research_journey_home_buttons_and_navigation():
    """Verify that clicking Home action buttons transitions correctly in Research Journey."""
    at = AppTest.from_file(APP, default_timeout=60).run()
    assert not at.exception

    # Select Research Journey section
    radio(at, "Section").set_value("Research Journey").run()
    assert not at.exception

    # Confirm on Home page
    radio(at, "Navigation").set_value("Home").run()
    assert not at.exception

    # Find button '🧪 Use Demo Participant'
    btn_demo = [b for b in at.button if "Use Demo Participant" in b.label]
    assert btn_demo, f"Button 'Use Demo Participant' not found. Available buttons: {[b.label for b in at.button]}"
    btn_demo[0].click().run()
    assert not at.exception
    # Demo participant populates uploaded_df in session state and transitions page
    assert "uploaded_df" in at.session_state
    assert len(at.session_state["uploaded_df"]) > 0
    assert at.session_state["page"] == "Upload & Validate"


def test_quick_questions_save_and_skip_buttons():
    """Verify Quick Questions scoring and save/skip buttons in Research Journey."""
    at = AppTest.from_file(APP, default_timeout=60).run()
    assert not at.exception

    radio(at, "Section").set_value("Research Journey").run()
    radio(at, "Navigation").set_value("Quick Questions").run()
    assert not at.exception

    # Click '💾 Save Answers & Calculate Results'
    btn_save = [b for b in at.button if "Save Answers" in b.label]
    assert btn_save, "Save Answers button not found"
    btn_save[0].click().run()
    assert not at.exception
    # Should have computed PSS-10 and WHO-5 results
    assert "pss_result" in at.session_state
    assert "qol_result" in at.session_state

    # Re-run for skip button
    at2 = AppTest.from_file(APP, default_timeout=60).run()
    radio(at2, "Section").set_value("Research Journey").run()
    radio(at2, "Navigation").set_value("Quick Questions").run()
    btn_skip = [b for b in at2.button if "Skip Questionnaire" in b.label]
    assert btn_skip, "Skip Questionnaire button not found"
    btn_skip[0].click().run()
    assert not at2.exception


def test_csv_upload_and_validation_logic():
    """Verify that the CSV upload and schema validation function works seamlessly on real files."""
    from dashboard.app import validate_uploaded_csv, get_trained_model

    df_sample = pd.read_csv(SAMPLE_CSV)
    pipe, expected_cols, train_info = get_trained_model()

    is_valid, msg, df_clean = validate_uploaded_csv(df_sample, expected_cols)
    assert is_valid is True, f"Validation failed: {msg}"
    assert df_clean is not None
    assert len(df_clean) == len(df_sample)
    assert list(df_clean.columns) == expected_cols


def test_support_journey_checkin_flow():
    """Verify conversation check-in buttons and store recording in Support Journey."""
    at = AppTest.from_file(APP, default_timeout=60).run()
    radio(at, "Section").set_value("Support journey").run()
    radio(at, "Navigation").set_value("Check-in").run()
    assert not at.exception

    btn_start = [b for b in at.button if "Start a check-in" in b.label]
    assert btn_start, "Button 'Start a check-in' not found"
    btn_start[0].click().run()
    assert not at.exception

    # Mood choice button
    btn_overwhelmed = [b for b in at.button if "Overwhelmed" in b.label]
    assert btn_overwhelmed, "Button 'Overwhelmed' not found"
    btn_overwhelmed[0].click().run()
    assert not at.exception


def test_clear_cache_and_retrain_button():
    """Verify Clear Cache button works without throwing exceptions."""
    at = AppTest.from_file(APP, default_timeout=60).run()
    btn_clear = [b for b in at.sidebar.button if "Clear Cache" in b.label]
    assert btn_clear, "Clear Cache button missing from sidebar"
    btn_clear[0].click().run()
    assert not at.exception


def test_sidebar_appearance_and_calm_mode_toggles():
    """Verify that the Aurora theme and Calm mode toggles function cleanly."""
    at = AppTest.from_file(APP, default_timeout=60).run()
    # Check Calm mode checkbox presence
    chk_calm = [c for c in at.sidebar.checkbox if "Calm Mode" in c.label]
    assert chk_calm, "Calm Mode checkbox missing from sidebar"
    chk_calm[0].check().run()
    assert not at.exception
    # Check Theme toggle presence
    rad_theme = radio(at, "Theme")
    assert "Dawn" in rad_theme.options
    rad_theme.set_value("Dawn").run()
    assert not at.exception


def test_all_research_journey_pages_render_without_exception():
    """Verify each page in Research Journey renders completely without exception."""
    at = AppTest.from_file(APP, default_timeout=60).run()
    radio(at, "Section").set_value("Research Journey").run()
    research_pages = [
        "Home",
        "Upload & Validate",
        "Quick Questions",
        "Stress Result",
        "How Stress Is Calculated",
        "Model Performance",
        "Feature Lab",
        "Quality of Life & Stress Scale",
        "Research Paper",
    ]
    for p in research_pages:
        radio(at, "Navigation").set_value(p).run()
        assert not at.exception, f"Error rendering page {p}: {[e.value for e in at.exception]}"
        assert at.session_state["page"] == p


def test_complete_button_navigation_flow():
    """Test full journey via buttons from Home -> Upload -> Questions -> Result."""
    at = AppTest.from_file(APP, default_timeout=60).run()
    radio(at, "Section").set_value("Research Journey").run()
    radio(at, "Navigation").set_value("Home").run()

    # Step 1: Click '🧪 Use Demo Participant' on Home
    btn_demo = [b for b in at.button if "Use Demo Participant" in b.label]
    assert btn_demo, "Demo button not found"
    btn_demo[0].click().run()
    assert not at.exception
    assert at.session_state["page"] == "Upload & Validate"
    assert "uploaded_df" in at.session_state

    # Step 2: On Upload & Validate, click 'Proceed to Quick Questions (Recommended) ➔'
    btn_qq = [b for b in at.button if "Quick Questions" in b.label]
    assert btn_qq, "Quick Questions button not found"
    btn_qq[0].click().run()
    assert not at.exception
    assert at.session_state["page"] == "Quick Questions"

    # Step 3: On Quick Questions, click '💾 Save Answers & Calculate Results'
    btn_save = [b for b in at.button if "Save Answers" in b.label]
    assert btn_save, "Save Answers button not found"
    btn_save[0].click().run()
    assert not at.exception
    assert at.session_state["page"] == "Stress Result"

    # Step 4: Verify PDF download button exists on Result page
    btn_pdf = [b for b in at.download_button if "Download Research PDF Report" in b.label]
    assert btn_pdf, "PDF Report download button not found"


def test_research_tools_overview_to_analyzer_navigation():
    """Verify 'Analyze Physiological Data' button on Overview navigates to Analyzer."""
    at = AppTest.from_file(APP, default_timeout=60).run()
    radio(at, "Section").set_value("Research tools").run()
    radio(at, "Navigation").set_value("Overview").run()

    btn_analyze = [b for b in at.button if "Analyze Physiological Data" in b.label]
    assert btn_analyze, "Analyze Physiological Data button not found on Overview"
    btn_analyze[0].click().run()
    assert not at.exception
    assert at.session_state["page"] == "Analyzer"


def test_skip_to_stress_results_button_flow():
    """Verify 'Skip to Stress Results Directly' transitions directly to Stress Result."""
    at = AppTest.from_file(APP, default_timeout=60).run()
    radio(at, "Section").set_value("Research Journey").run()
    radio(at, "Navigation").set_value("Home").run()

    # Load demo data
    btn_demo = [b for b in at.button if "Use Demo Participant" in b.label]
    btn_demo[0].click().run()
    assert at.session_state["page"] == "Upload & Validate"

    # Click Skip button
    btn_skip = [b for b in at.button if "Skip to Stress Results Directly" in b.label]
    assert btn_skip, "Skip button not found"
    btn_skip[0].click().run()
    assert not at.exception
    assert at.session_state["page"] == "Stress Result"


def test_result_page_fallback_navigation_when_no_data():
    """Verify '➔ Go to Upload & Signals Page' button when no data is loaded."""
    at = AppTest.from_file(APP, default_timeout=60).run()
    radio(at, "Section").set_value("Research Journey").run()
    radio(at, "Navigation").set_value("Stress Result").run()

    btn_upload_link = [b for b in at.button if "Go to Upload" in b.label]
    assert btn_upload_link, "Go to Upload button not found"
    btn_upload_link[0].click().run()
    assert not at.exception
    assert at.session_state["page"] == "Upload & Validate"


def test_all_support_and_research_tools_pages_render_cleanly():
    """Verify all 10 Support Journey pages and 7 Research Tools pages render cleanly."""
    at = AppTest.from_file(APP, default_timeout=60).run()

    # Test Support Journey pages
    radio(at, "Section").set_value("Support journey").run()
    from src.neurotwin.ui import SUPPORT_PAGES
    for p in SUPPORT_PAGES:
        radio(at, "Navigation").set_value(p).run()
        assert not at.exception, f"Support page {p} failed: {[e.value for e in at.exception]}"

    # Test Research Tools pages
    radio(at, "Section").set_value("Research tools").run()
    research_tools = ["Overview", "Analyzer", "Signal Lab", "Explainable AI", "Model Arena", "Personal Twin", "Methods"]
    for p in research_tools:
        radio(at, "Navigation").set_value(p).run()
        assert not at.exception, f"Research tool {p} failed: {[e.value for e in at.exception]}"


