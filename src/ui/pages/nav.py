"""Shared navigation registry: canonical st.Page objects for the app.

Pages are built lazily so page modules can import this module without
creating import cycles. ``st.switch_page`` and ``st.page_link`` need the
actual ``Page`` objects (string paths only resolve for file-based pages),
so every cross-page jump goes through ``get_pages()``.
"""

from __future__ import annotations

import streamlit as st


def get_pages() -> dict[str, st.Page]:
    """The canonical page objects, keyed by their ``url_path``."""
    from . import (
        about,
        analytics,
        compare,
        dashboard,
        evidence,
        findings,
        methodology,
    )

    return {
        "dashboard": st.Page(
            dashboard.render,
            title="Dashboard",
            icon="📈",
            url_path="dashboard",
            default=True,
        ),
        "analytics": st.Page(
            analytics.render, title="Analytics", icon="📊", url_path="analytics"
        ),
        "compare": st.Page(
            compare.render, title="Compare", icon="⚖️", url_path="compare"
        ),
        "findings": st.Page(
            findings.render, title="Findings", icon="🔬", url_path="findings"
        ),
        "evidence": st.Page(
            evidence.render, title="Evidence", icon="🧪", url_path="evidence"
        ),
        "methodology": st.Page(
            methodology.render, title="Methodology", icon="📚", url_path="methodology"
        ),
        "about": st.Page(about.render, title="About", icon="🎓", url_path="about"),
    }
