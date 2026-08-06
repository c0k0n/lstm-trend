"""AppTest end-to-end tests for the Streamlit app (network access required)."""

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

APP_PATH = Path(__file__).resolve().parent.parent / "streamlit_app.py"


@pytest.fixture()
def app():
    at = AppTest.from_file(str(APP_PATH), default_timeout=600)
    at.run()
    return at


@pytest.mark.e2e
def test_dashboard_renders(app):
    assert not app.exception
    assert "LSTM Trend" in app.title[0].value


@pytest.mark.e2e
def test_run_full_analysis(app):
    app.sidebar.button[0].click().run()
    assert not app.exception
    # Pipeline finished: final status visible, metrics rendered
    statuses = [s.label for s in app.status]
    assert any("complete" in label for label in statuses)
    assert len(app.metric) > 0


@pytest.mark.e2e
def test_forecast_table_and_download(app):
    app.sidebar.button[0].click().run()
    assert not app.exception
    assert any(b.label.endswith("Download forecast (CSV)") for b in app.download_button)


@pytest.mark.e2e
def test_findings_page_shows_comparison(app):
    app.sidebar.button[0].click().run()
    app.switch_page("src/ui/pages/findings.py").run()
    assert not app.exception
    assert any(h.value == "How the models compare" for h in app.subheader)


@pytest.mark.e2e
def test_device_caption_present(app):
    assert any(c.value.startswith("Training device:") for c in app.sidebar.caption)


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
