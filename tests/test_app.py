"""AppTest end-to-end tests for the Streamlit app (network access required)."""

import importlib.util
from pathlib import Path

import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest

from src.constants import GITHUB_URL

APP_PATH = Path(__file__).resolve().parent.parent / "streamlit_app.py"


@pytest.fixture()
def app():
    at = AppTest.from_file(str(APP_PATH), default_timeout=600)
    at.run()
    return at


def _click_run(at: AppTest) -> None:
    run_button = next(b for b in at.sidebar.button if "Run analysis" in b.label)
    run_button.click().run()


@pytest.mark.e2e
def test_dashboard_renders(app):
    assert not app.exception
    assert "LSTM Trend" in app.title[0].value


@pytest.mark.e2e
def test_footer_links_to_repo(monkeypatch):
    class _NoopNav:
        def run(self):
            return None

    captured: dict[str, str] = {}
    monkeypatch.setattr(st, "set_page_config", lambda **kwargs: None)
    monkeypatch.setattr(st, "logo", lambda *args, **kwargs: None)
    monkeypatch.setattr(st, "navigation", lambda *args, **kwargs: _NoopNav())
    monkeypatch.setattr(st, "html", lambda html: captured.setdefault("html", html))

    spec = importlib.util.spec_from_file_location("streamlit_app_entry", str(APP_PATH))
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    assert GITHUB_URL in captured["html"]


@pytest.mark.e2e
def test_run_full_analysis(app):
    _click_run(app)
    assert not app.exception
    # Pipeline finished: final status visible, metrics rendered
    statuses = [s.label for s in app.status]
    assert any("complete" in label for label in statuses)
    assert len(app.metric) > 0


@pytest.mark.e2e
def test_forecast_table_and_download(app):
    _click_run(app)
    assert not app.exception
    assert any(b.label.endswith("Download forecast (CSV)") for b in app.download_button)


@pytest.mark.e2e
def test_findings_page_shows_comparison(app):
    _click_run(app)
    app.switch_page("src/ui/pages/findings.py").run()
    assert not app.exception
    assert any(h.value == "How the models compare" for h in app.subheader)


@pytest.mark.e2e
def test_device_caption_present(app):
    assert any(c.value.startswith("Training device:") for c in app.sidebar.caption)


@pytest.mark.e2e
def test_sidebar_settings_caption(app):
    captions = [c.value for c in app.sidebar.caption]
    assert any("Current settings:" in c for c in captions)


@pytest.mark.e2e
def test_sidebar_model_settings_form(app):
    assert any(b.label == "Apply model settings" for b in app.sidebar.button)
    horizons = [
        s for s in app.segmented_control if s.label == "Forecast horizon (days)"
    ]
    assert horizons and horizons[0].options == ["5", "15", "30", "Custom…"]


@pytest.mark.e2e
def test_query_param_deep_links():
    at = AppTest.from_file(str(APP_PATH), default_timeout=600)
    at.query_params["ticker"] = "MSFT"
    at.query_params["start"] = "2024-01-01"
    at.query_params["end"] = "2025-01-01"
    at.query_params["horizon"] = "30"
    at.run()
    assert not at.exception
    assert at.session_state["ticker_custom"] == "MSFT"
    assert at.session_state["start_date"].isoformat() == "2024-01-01"
    assert at.session_state["end_date"].isoformat() == "2025-01-01"
    assert at.session_state["horizon_preset"] == 30


@pytest.mark.e2e
def test_query_param_custom_horizon_deep_link():
    at = AppTest.from_file(str(APP_PATH), default_timeout=600)
    at.query_params["horizon"] = "21"
    at.run()
    assert not at.exception
    assert at.session_state["horizon_preset"] == "Custom…"
    assert at.session_state["fut_steps"] == 21


@pytest.mark.e2e
def test_query_param_custom_horizon_clamped():
    at = AppTest.from_file(str(APP_PATH), default_timeout=600)
    at.query_params["horizon"] = "999"
    at.run()
    assert not at.exception
    assert at.session_state["horizon_preset"] == "Custom…"
    assert at.session_state["fut_steps"] == 90


@pytest.mark.e2e
def test_analytics_page_lazy_tabs(app):
    app.switch_page("src/ui/pages/analytics.py").run()
    assert not app.exception
    assert [t.label for t in app.tabs] == [
        "📈 Overview",
        "📊 Returns",
        "📅 Seasonality",
        "🧭 Technicals",
        "🕳️ Risk",
    ]
    assert len(app.tabs[0].metric) > 0


@pytest.mark.e2e
def test_findings_verdict_feedback(app):
    _click_run(app)
    app.switch_page("src/ui/pages/findings.py").run()
    assert not app.exception
    assert len(app.feedback) >= 1


@pytest.mark.e2e
def test_horizon_form_apply_reflected_in_url(app):
    horizon = next(
        s for s in app.segmented_control if s.label == "Forecast horizon (days)"
    )
    horizon.set_value(30)
    submit = next(b for b in app.sidebar.button if b.label == "Apply model settings")
    submit.click().run()
    assert not app.exception
    _click_run(app)
    assert not app.exception
    assert app.query_params["horizon"] == ["30"]


@pytest.mark.e2e
def test_analytics_page_renders(app):
    app.switch_page("src/ui/pages/analytics.py").run()
    assert not app.exception
    assert any(t.value == "📊 Analytics" for t in app.title)
    # Default ticker loads data and shows the overview tab
    assert len(app.metric) > 0


@pytest.mark.e2e
def test_compare_page_renders(app):
    app.switch_page("src/ui/pages/compare.py").run()
    assert not app.exception
    assert any(t.value == "⚖️ Compare" for t in app.title)
    # Default three tickers produce a metrics table
    assert len(app.dataframe) >= 1
