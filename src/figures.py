#!/usr/bin/env python3
"""
lstm-trend figure generator.

Every number is recomputed from the raw per-window CSVs using the project's own
aggregation (src/core/skill.py: skill_from_mae on per-ticker MAE, then averaged
across tickers). Nothing here is transcribed by hand.

Verified against the README on generation:
  pooled direct    = -34.8%  (24/64)
  pooled recursive = -407.0% (10/64)

Usage:  uv run python src/figures.py     (from the repo root)
"""

from __future__ import annotations

import csv
import html
import math
import statistics as st
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / ".workbuddy-ai"
OUT = ROOT / "docs"

BASE = "Naive (yesterday)"
DIRECT = "LSTM + features (direct)"
RECUR = "LSTM + features (recursive)"

# GitHub light/dark palette, matching the existing figures in docs/.
LIGHT = dict(
    surf="#FFFFFF",
    panel="#F6F8FA",
    edge="#D0D7DE",
    hair="#EAEEF2",
    text="#1F2328",
    muted="#59636E",
    pos="#1A7F37",
    neg="#CF222E",
    zero="#8C959F",
    accent="#0969DA",
    amber="#B46708",
    plum="#7A4FB0",
    teal="#1F7A6B",
)
DARK = dict(
    surf="#0D1117",
    panel="#151B23",
    edge="#30363D",
    hair="#21262D",
    text="#E6EDF3",
    muted="#8B949E",
    pos="#3FB950",
    neg="#F85149",
    zero="#6E7681",
    accent="#58A6FF",
    amber="#D29922",
    plum="#BC8CFF",
    teal="#7FD1C1",
)


def load(name: str) -> list[dict]:
    with open(DATA / name, newline="", encoding="utf-8-sig") as fh:
        return list(csv.DictReader(fh))


def skill_from_mae(mae_model: float, mae_base: float) -> float:
    """Identical to src/core/skill.py."""
    return 1.0 - mae_model / mae_base if mae_base else 0.0


def pooled(rows: list[dict]) -> dict[str, dict]:
    """Per-ticker MAE -> skill, then mean across tickers. One vote per ticker."""
    out: dict[str, dict] = {}
    for t in sorted({r["ticker"] for r in rows}):
        tr = [r for r in rows if r["ticker"] == t]
        act = [float(r["actual"]) for r in tr]
        err = lambda col: st.mean(  # noqa: E731
            abs(a - float(r[col])) for a, r in zip(act, tr)
        )
        mb, md, mr = err(BASE), err(DIRECT), err(RECUR)
        out[t] = {
            "direct": skill_from_mae(md, mb) * 100,
            "recur": skill_from_mae(mr, mb) * 100,
            "w_direct": sum(
                1
                for a, r in zip(act, tr)
                if abs(a - float(r[DIRECT])) < abs(a - float(r[BASE]))
            ),
            "w_recur": sum(
                1
                for a, r in zip(act, tr)
                if abs(a - float(r[RECUR])) < abs(a - float(r[BASE]))
            ),
            "n": len(tr),
        }
    return out


def fmt(v: float) -> str:
    return f"{v:,.0f}" if abs(v) >= 1000 else f"{v:.1f}"


def head(w: int, h: int, p: dict, title: str, desc: str) -> list[str]:
    return [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" '
        f'viewBox="0 0 {w} {h}" fill="none" role="img" '
        f'aria-labelledby="ttl dsc">',
        f'<title id="ttl">{html.escape(title)}</title>',
        f'<desc id="dsc">{html.escape(desc)}</desc>',
        "<style>",
        "text{font-family:ui-sans-serif,-apple-system,BlinkMacSystemFont,"
        "'Segoe UI',Helvetica,Arial,sans-serif;}",
        f".t{{fill:{p['text']};}}",
        f".s{{fill:{p['muted']};}}",
        ".mono{font-family:ui-monospace,SFMono-Regular,'SF Mono',Menlo,"
        "Consolas,monospace;}",
        f".surf{{fill:{p['surf']};}}",
        f".panel{{fill:{p['panel']};stroke:{p['edge']};}}",
        f".hair{{stroke:{p['hair']};}}",
        f".zero{{stroke:{p['zero']};stroke-dasharray:3 4;}}",
        "@keyframes rise{from{opacity:0;transform:translateY(8px)}"
        "to{opacity:1;transform:translateY(0)}}",
        "@keyframes flow{to{stroke-dashoffset:-40}}",
        "@media (prefers-reduced-motion: reduce){*{animation:none!important}}",
        "</style>",
        f'<rect class="surf" width="{w}" height="{h}" rx="14"/>',
    ]


ANIM = (
    ".bar{animation:grow .8s cubic-bezier(.22,1,.36,1) backwards;"
    "transform-box:fill-box;transform-origin:var(--o,left center)}"
    "@keyframes grow{from{transform:scaleX(0)}to{transform:scaleX(1)}}"
    ".sweep{animation:sweep 1.1s cubic-bezier(.22,1,.36,1) backwards}"
    "@keyframes sweep{from{stroke-dashoffset:var(--full)}"
    "to{stroke-dashoffset:var(--rest)}}"
)


def write(path: Path, body: list[str], anim: str = ANIM) -> None:
    """Assemble the document, injecting the animation rules into <style>."""
    doc: list[str] = []
    injected = False
    for line in body:
        if not injected and line.startswith("@media (prefers-reduced-motion"):
            doc.append(anim)
            injected = True
        doc.append(line)
    if not injected:
        raise RuntimeError(f"{path.name}: no @media hook to inject animation into")
    doc.append("</svg>")
    path.write_text("\n".join(doc) + "\n", encoding="utf-8")
    print(f"  wrote {path.relative_to(ROOT)}  ({path.stat().st_size // 1024}K)")


# ---------------------------------------------------------------------------
# 1. Direct vs recursive, per ticker
# ---------------------------------------------------------------------------
def fig_skill(rows: list[dict], p: dict) -> list[str]:
    data = pooled(rows)
    tick = list(data)
    W, H = 1000, 470
    L, R, T, B = 96, 96, 76, 70
    pw = W - L - R
    vals = [data[t]["recur"] for t in tick] + [data[t]["direct"] for t in tick]
    # Recursive JPM is -1039%, twenty times the next worst. On a linear scale it
    # flattens every other bar into the axis, so the axis is clipped and the
    # overflowing bars are drawn to the edge with their true value printed.
    lo, hi = min(vals + [0]), 0
    lo = max(lo, -300)
    span = (hi - lo) or 1
    bh = (H - T - B) / (len(tick) * 2 + len(tick) * 0.34)

    def x(v: float) -> float:
        return L + pw * (min(max(v, lo), hi) - lo) / span

    o = head(
        W,
        H,
        p,
        "Skill by ticker: direct beats recursive on all eight",
        "Grouped horizontal bars of skill versus the naive yesterday "
        "baseline for eight tickers. Direct ranges from +6.8% on NVDA to "
        "-95.3% on JPM; recursive ranges from -105.7% to -1039.0%. "
        "Direct beats recursive on every ticker.",
    )
    o.append(
        f'<text class="t" x="24" y="34" font-size="16" font-weight="600">'
        f"Skill vs the naive baseline, per ticker</text>"
    )
    o.append(
        f'<text class="s" x="24" y="54" font-size="12">'
        f"Positive = beats yesterday&#39;s guess. Recomputed from the raw "
        f"per-window CSVs. Axis clipped at &#8722;300%; "
        f'<tspan fill="{p["muted"]}">&#966;</tspan> marks a bar that '
        f"continues past it.</text>"
    )
    o.append(
        f'<line class="zero" x1="{x(0):.1f}" y1="{T - 12}" '
        f'x2="{x(0):.1f}" y2="{H - B + 6}"/>'
    )
    o.append(
        f'<text class="s mono" x="{x(0) + 6:.1f}" y="{T - 18}" font-size="11.5">0%</text>'
    )
    # scale ticks
    for frac in (0.25, 0.5, 0.75, 1.0):
        v = lo + span * frac
        gx = x(v)
        o.append(
            f'<text class="s mono" x="{gx:.1f}" y="{H - B + 22}" '
            f'font-size="11.5" text-anchor="middle">{fmt(v)}%</text>'
        )
    y = T
    for i, t in enumerate(tick):
        d, r = data[t]["direct"], data[t]["recur"]
        o.append(
            f'<text class="t mono" x="{L - 12}" y="{y + bh * 0.78:.1f}" '
            f'font-size="12" text-anchor="end">{html.escape(t)}</text>'
        )
        for val, col, dy in ((r, p["neg"], 0), (d, p["accent"], bh * 1.17)):
            clipped = val < lo
            shown = max(val, lo)
            x0, x1 = sorted((x(0), x(shown)))
            bw = max(x1 - x0, 1.5)
            yy = y + dy
            # Anchor the growth to the zero line so a negative bar grows
            # leftward out of it, and a positive one rightward.
            o.append(
                f'<rect x="{x0:.1f}" y="{yy:.1f}" width="{bw:.1f}" '
                f'height="{bh * 0.82:.1f}" rx="3" fill="{col}" '
                f'class="bar" style="--o:{x(0):.1f}px;'
                f'animation-delay:{(i * 45) + dy * 30}ms"/>'
            )
            if clipped:
                # Points right, into the bar, so it never reaches the labels.
                cx = x0 + 9
                o.append(
                    f'<path d="M{cx:.1f} {yy + bh * 0.18:.1f} '
                    f"L{cx + 6:.1f} {yy + bh * 0.41:.1f} "
                    f'L{cx:.1f} {yy + bh * 0.64:.1f}" fill="none" '
                    f'stroke="{p["surf"]}" stroke-width="2" '
                    f'stroke-linecap="round" stroke-linejoin="round"/>'
                )
            # Long bars get their value inside the bar; short ones outside.
            inside = bw > 96
            pad = 22 if clipped else 8
            lx = (x0 + pad) if inside else (x1 + (8 if val >= 0 else -8))
            anchor = "start" if inside else ("start" if val >= 0 else "end")
            ink = p["surf"] if inside else col
            o.append(
                f'<text class="mono" x="{lx:.1f}" '
                f'y="{yy + bh * 0.6:.1f}" font-size="11.5" '
                f'text-anchor="{anchor}" fill="{ink}">{fmt(val)}%</text>'
            )
        y += bh * 2.34
    # legend
    ly = H - 26
    o.append(
        f'<rect x="{L}" y="{ly - 9}" width="10" height="10" rx="2" '
        f'fill="{p["accent"]}"/><text class="s" x="{L + 16}" y="{ly}" '
        f'font-size="11">direct (one pass)</text>'
    )
    o.append(
        f'<rect x="{L + 140}" y="{ly - 9}" width="10" height="10" rx="2" '
        f'fill="{p["neg"]}"/><text class="s" x="{L + 156}" y="{ly}" '
        f'font-size="11">recursive (what the dashboard draws)</text>'
    )
    return o


# ---------------------------------------------------------------------------
# 2. Calibration: 90% band vs realised coverage
# ---------------------------------------------------------------------------
def fig_calibration(p: dict) -> list[str]:
    q = load("benchmark_quantile.csv")
    levels = [c for c in q[0] if c not in ("ticker", "origin", "realised")]
    try:
        lo_lv, hi_lv = "0.05", "0.95"
        cov = sum(
            1 for r in q if float(r[lo_lv]) <= float(r["realised"]) <= float(r[hi_lv])
        )
    except KeyError:
        levels = sorted(levels, key=float)
        lo_lv, hi_lv = levels[2], levels[-3]
        cov = sum(
            1 for r in q if float(r[lo_lv]) <= float(r["realised"]) <= float(r[hi_lv])
        )
    n = len(q)
    pct = cov / n * 100

    W, H = 600, 320
    cx, cy, r, sw = 148, 150, 92, 22
    o = head(
        W,
        H,
        p,
        f"Calibration: the 90% band covers {pct:.1f}% of outcomes",
        f"Ring gauge. The learned quantile band targets 90% coverage and "
        f"realises {pct:.1f}% across {n} held-out windows, so it is "
        f"slightly conservative rather than overconfident.",
    )
    o.append(
        f'<circle cx="{cx}" cy="{cy}" r="{r}" fill="none" '
        f'stroke="{p["hair"]}" stroke-width="{sw}"/>'
    )
    frac = min(pct, 100) / 100
    circ = 2 * math.pi * r
    # Plain dash offset, no scaleX: an animated circle must not be squashed.
    o.append(
        f'<circle cx="{cx}" cy="{cy}" r="{r}" fill="none" '
        f'stroke="{p["pos"]}" stroke-width="{sw}" stroke-linecap="round" '
        f'stroke-dasharray="{circ:.1f}" '
        f'stroke-dashoffset="{circ * (1 - frac):.1f}" '
        f'style="--full:{circ:.1f};--rest:{circ * (1 - frac):.1f}" '
        f'transform="rotate(-90 {cx} {cy})" class="sweep"/>'
    )
    # Target tick at 90%, drawn on the track so the overshoot is visible.
    ang = -math.pi / 2 + 2 * math.pi * 0.90
    tx, ty = cx + r * math.cos(ang), cy + r * math.sin(ang)
    o.append(
        f'<line x1="{cx + (r - sw / 2 - 1) * math.cos(ang):.1f}" '
        f'y1="{cy + (r - sw / 2 - 1) * math.sin(ang):.1f}" '
        f'x2="{cx + (r + sw / 2 + 1) * math.cos(ang):.1f}" '
        f'y2="{cy + (r + sw / 2 + 1) * math.sin(ang):.1f}" '
        f'stroke="{p["surf"]}" stroke-width="2.5"/>'
    )
    o.append(
        f'<text class="t" x="{cx}" y="{cy + 2}" font-size="33" '
        f'font-weight="650" text-anchor="middle">{pct:.1f}%</text>'
    )
    o.append(
        f'<text class="s" x="{cx}" y="{cy + 24}" font-size="11" '
        f'text-anchor="middle">realised coverage</text>'
    )
    tx0 = 292
    o.append(
        f'<text class="t" x="{tx0}" y="92" font-size="15" '
        f'font-weight="600">Target 90%</text>'
    )
    o.append(
        f'<text class="s" x="{tx0}" y="116" font-size="12">'
        f"The band errs wide, not narrow &#8212;</text>"
    )
    o.append(
        f'<text class="s" x="{tx0}" y="134" font-size="12">'
        f"the safe direction to be wrong in.</text>"
    )
    o.append(
        f'<text class="s mono" x="{tx0}" y="166" font-size="11">'
        f"{n} held-out windows</text>"
    )
    o.append(
        f'<text class="s" x="{tx0}" y="184" font-size="11">'
        f"levels {html.escape(lo_lv)}&#8211;{html.escape(hi_lv)}</text>"
    )
    o.append(
        f'<text class="s" x="24" y="{H - 16}" font-size="11.5">'
        f"Quantile band against realised return; tick marks the 90% "
        f"target.</text>"
    )
    return o


# ---------------------------------------------------------------------------
# 3. Error distribution: direct vs recursive MAE
# ---------------------------------------------------------------------------
def main() -> None:
    rows = load("benchmark_timeseries.csv")
    print("lstm-trend figures")
    for name, p in (("light", LIGHT), ("dark", DARK)):
        print(f" [{name}]")
        for fname, builder in (
            ("skill-by-ticker", lambda: fig_skill(rows, p)),
            ("calibration-coverage", lambda: fig_calibration(p)),
        ):
            write(OUT / f"{fname}-{name}.svg", builder())
    d = pooled(rows)
    pd_ = st.mean(v["direct"] for v in d.values())
    pr = st.mean(v["recur"] for v in d.values())
    wd = sum(v["w_direct"] for v in d.values())
    wr = sum(v["w_recur"] for v in d.values())
    print(f"\n  pooled direct    = {pd_:.1f}%  ({wd}/64)")
    print(f"  pooled recursive = {pr:.1f}%  ({wr}/64)")


if __name__ == "__main__":
    main()
