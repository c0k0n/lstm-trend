"""Guards for the SVG figures in docs/.

Every one of these checks exists because the diagram passed a "does it render?"
test and failed a "can you read it?" test. Validity is not legibility, and an
SVG that renders can still be wrong in ways only coordinates reveal:

  * a connector that stops in empty space, or runs through a box
  * ink the same colour as the page it is drawn on
  * text small enough to vanish once the README scales the image down
  * marker definitions nothing references

Charts (data-split, forecast-drift) are exempt from the routing check only;
their "boxes" are bars and band fills, which lines are meant to cross.
"""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

NS = "{http://www.w3.org/2000/svg}"
DOCS = Path(__file__).resolve().parents[1] / "docs"
MIN_FONT_SIZE = 11.0

# diagrams whose lines are data flow, so a line through a box is always a bug
FLOW_DIAGRAMS = {"lstm-cell.svg", "lstm-stack.svg", "quantile-stack.svg"}


@pytest.fixture(scope="module")
def svgs() -> dict[str, ET.Element]:
    return {p.name: ET.parse(p).getroot() for p in sorted(DOCS.glob("*.svg"))}


def _attr(el: ET.Element, name: str) -> str:
    """A required attribute: absent means the figure is malformed, not empty."""
    value = el.get(name)
    assert value is not None, f"<{el.tag.replace(NS, '')}> is missing {name}"
    return value


def _num(el: ET.Element, name: str) -> float:
    return float(_attr(el, name))


def _viewbox(root: ET.Element) -> tuple[float, float]:
    _, _, w, h = (float(v) for v in _attr(root, "viewBox").split())
    return w, h


def _font_sizes(root: ET.Element) -> list[float]:
    """Declared sizes, from attributes and from CSS in any <style> block."""
    sizes = [
        _num(el, "font-size")
        for el in root.iter()
        if el.tag.replace(NS, "") == "text" and el.get("font-size")
    ]
    for style in root.iter(f"{NS}style"):
        for shorthand in re.findall(r"font:\s*[^;]*?([\d.]+)px", style.text or ""):
            sizes.append(float(shorthand))
        for longhand in re.findall(r"font-size:\s*([\d.]+)px", style.text or ""):
            sizes.append(float(longhand))
    return sizes


def _css_fills(root: ET.Element) -> dict[str, str]:
    """class name -> fill, from `fill:` declarations inside any <style> block."""
    fills: dict[str, str] = {}
    for style in root.iter(f"{NS}style"):
        for selector, body in re.findall(r"\.(\w+)\s*\{([^}]*)\}", style.text or ""):
            match = re.search(r"fill:\s*([^;]+)", body)
            if match:
                fills[selector] = match.group(1).strip()
    return fills


def _fill_of(el: ET.Element, css: dict[str, str]) -> str:
    """Inline fill wins; otherwise whatever the element's class declares."""
    inline = (el.get("fill") or "").strip()
    if inline:
        return inline
    for class_name in (el.get("class") or "").split():
        if class_name in css:
            return css[class_name]
    return ""


def _segments(root: ET.Element) -> list[tuple[float, float, float, float]]:
    """Every straight run in a <line> or an H/V <path>."""
    out: list[tuple[float, float, float, float]] = []
    for el in root.iter():
        tag = el.tag.replace(NS, "")
        if tag == "line":
            out.append((_num(el, "x1"), _num(el, "y1"), _num(el, "x2"), _num(el, "y2")))
        elif tag == "path":
            tokens = re.findall(r"[MHV]|-?\d+\.?\d*", el.get("d", ""))
            x = y = 0.0
            i = 0
            while i < len(tokens):
                cmd = tokens[i]
                if cmd == "M":
                    x, y = float(tokens[i + 1]), float(tokens[i + 2])
                    i += 3
                elif cmd == "H":
                    nx = float(tokens[i + 1])
                    out.append((x, y, nx, y))
                    x = nx
                    i += 2
                elif cmd == "V":
                    ny = float(tokens[i + 1])
                    out.append((x, y, x, ny))
                    y = ny
                    i += 2
                else:
                    i += 1
    return out


def _boxes(
    root: ET.Element, w: float, h: float
) -> list[tuple[float, float, float, float]]:
    """Node rectangles: everything except the page-sized background."""
    out = []
    for el in root.iter():
        if el.tag.replace(NS, "") != "rect":
            continue
        bw, bh = float(el.get("width") or 0), float(el.get("height") or 0)
        if bw >= w or bh >= h:
            continue
        x, y = float(el.get("x", 0)), float(el.get("y", 0))
        out.append((x, y, x + bw, y + bh))
    return out


def test_every_diagram_parses(svgs: dict[str, ET.Element]) -> None:
    assert svgs, f"no SVG figures found in {DOCS}"


def test_background_is_opaque(svgs: dict[str, ET.Element]) -> None:
    """Transparent ink on a transparent page is invisible on GitHub light."""
    for name, root in svgs.items():
        w, h = _viewbox(root)
        css = _css_fills(root)
        opaque = any(
            el.tag.replace(NS, "") == "rect"
            and float(el.get("width", 0)) >= w
            and float(el.get("height", 0)) >= h
            and _fill_of(el, css).lower() not in {"", "none", "transparent"}
            for el in root.iter()
        )
        assert opaque, f"{name} has no full-bleed opaque background rect"


def test_text_is_large_enough_to_read(svgs: dict[str, ET.Element]) -> None:
    for name, root in svgs.items():
        sizes = _font_sizes(root)
        assert sizes, f"{name} declares no font size anywhere"
        smallest = min(sizes)
        assert smallest >= MIN_FONT_SIZE, f"{name} has {smallest}px text"


def test_no_connector_runs_through_a_node(svgs: dict[str, ET.Element]) -> None:
    for name, root in svgs.items():
        if name not in FLOW_DIAGRAMS:
            continue
        w, h = _viewbox(root)
        boxes = _boxes(root, w, h)
        for ax, ay, bx, by in _segments(root):
            lo_x, hi_x = sorted((ax, bx))
            lo_y, hi_y = sorted((ay, by))
            vertical = abs(ax - bx) < 1e-9
            horizontal = abs(ay - by) < 1e-9
            if not (vertical or horizontal):
                continue
            for x0, y0, x1, y1 in boxes:
                if vertical:
                    through = x0 < ax < x1
                    depth = min(hi_y, y1) - max(lo_y, y0)
                else:
                    through = y0 < ay < y1
                    depth = min(hi_x, x1) - max(lo_x, x0)
                assert not (through and depth > 1.0), (
                    f"{name}: connector ({ax:.0f},{ay:.0f})->({bx:.0f},{by:.0f}) "
                    f"runs {depth:.0f}px through the box at ({x0:.0f},{y0:.0f})"
                )


def test_markers_are_defined_and_used(svgs: dict[str, ET.Element]) -> None:
    """A marker nothing references is dead weight; a missing one is an invisible arrow."""
    for name, root in svgs.items():
        defined: set[str] = set()
        used: set[str] = set()
        for el in root.iter():
            marker_id = el.get("id")
            if el.tag.replace(NS, "") == "marker" and marker_id is not None:
                defined.add(marker_id)
        for el in root.iter():
            for value in el.attrib.values():
                used.update(re.findall(r"url\(#([^)]+)\)", value))
        missing = used - defined
        assert not missing, f"{name} references undefined marker(s): {sorted(missing)}"
        assert not defined - used, (
            f"{name} defines unused marker(s): {sorted(defined - used)}"
        )


def test_nothing_drawn_outside_the_canvas(svgs: dict[str, ET.Element]) -> None:
    for name, root in svgs.items():
        w, h = _viewbox(root)
        for el in root.iter():
            tag = el.tag.replace(NS, "")
            points: list[tuple[float, float]] = []
            if tag == "text":
                points.append((_num(el, "x"), _num(el, "y")))
            elif tag == "circle":
                points.append((_num(el, "cx"), _num(el, "cy")))
            elif tag == "rect":
                x, y = float(el.get("x") or 0), float(el.get("y") or 0)
                points.append((x, y))
                points.append(
                    (x + float(el.get("width") or 0), y + float(el.get("height") or 0))
                )
            for px, py in points:
                assert 0 <= px <= w and 0 <= py <= h, (
                    f"{name}: {tag} at ({px:.0f},{py:.0f}) is outside {w:.0f}x{h:.0f}"
                )
