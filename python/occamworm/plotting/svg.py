"""Dependency-free SVG figures for reports (OW-015). Every plotted value is passed in; nothing is computed here."""

from __future__ import annotations

from collections.abc import Sequence
from html import escape

W, H, PAD = 640, 28, 150


def forest(rows: Sequence[tuple[str, float, float, float]], title: str, xlabel: str) -> str:
    """Point estimates with intervals: rows of (label, estimate, low, high). A dashed line marks zero."""
    if not rows:
        return ""
    lo = min(min(r[2] for r in rows), 0.0)
    hi = max(max(r[3] for r in rows), 0.0)
    span = (hi - lo) or 1.0
    width = W - PAD - 20

    def x(v: float) -> float:
        return PAD + (v - lo) / span * width

    height = 50 + H * len(rows)
    out = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{height}" font-family="sans-serif" '
        'font-size="11">',
        f'<text x="{W / 2}" y="16" text-anchor="middle" font-size="13">{escape(title)}</text>',
        f'<line x1="{x(0):.1f}" y1="24" x2="{x(0):.1f}" y2="{height - 22}" stroke="#999" stroke-dasharray="4"/>',
    ]
    for i, (label, est, a, b) in enumerate(rows):
        y = 36 + H * i
        out += [
            f'<text x="{PAD - 8}" y="{y + 4}" text-anchor="end">{escape(label)}</text>',
            f'<line x1="{x(a):.1f}" y1="{y}" x2="{x(b):.1f}" y2="{y}" stroke="#333"/>',
            f'<circle cx="{x(est):.1f}" cy="{y}" r="3.5" fill="#1f5fa8"/>',
        ]
    out += [
        f'<text x="{x(lo):.1f}" y="{height - 6}" text-anchor="middle">{lo:.3g}</text>',
        f'<text x="{x(hi):.1f}" y="{height - 6}" text-anchor="middle">{hi:.3g}</text>',
        f'<text x="{PAD + width / 2}" y="{height - 6}" text-anchor="middle">{escape(xlabel)}</text>',
        "</svg>",
    ]
    return "\n".join(out) + "\n"


def lines(series: Sequence[tuple[str, Sequence[float], Sequence[float]]], title: str, xlabel: str) -> str:
    """Line plot of (label, x, y) series."""
    xs = [v for _, xx, _ in series for v in xx]
    ys = [v for _, _, yy in series for v in yy]
    if not xs:
        return ""
    x0, x1, y0, y1 = min(xs), max(xs), min(ys), max(ys)
    sx, sy = (x1 - x0) or 1.0, (y1 - y0) or 1.0
    width, height = W - 80, 260
    palette = ["#1f5fa8", "#c0392b", "#27ae60", "#8e44ad", "#d35400", "#2c3e50"]
    out = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{height + 60}" font-family="sans-serif" '
        'font-size="11">',
        f'<text x="{W / 2}" y="16" text-anchor="middle" font-size="13">{escape(title)}</text>',
    ]
    for k, (label, xx, yy) in enumerate(series):
        pts = " ".join(
            f"{60 + (a - x0) / sx * width:.1f},{30 + height - (b - y0) / sy * height:.1f}"
            for a, b in zip(xx, yy, strict=True)
        )
        col = palette[k % len(palette)]
        out += [
            f'<polyline fill="none" stroke="{col}" points="{pts}"/>',
            f'<text x="{W - 10}" y="{30 + 14 * k}" text-anchor="end" fill="{col}">{escape(label)}</text>',
        ]
    out += [
        f'<text x="{60 + width / 2}" y="{height + 52}" text-anchor="middle">{escape(xlabel)}</text>',
        f'<text x="56" y="{30 + height}" text-anchor="end">{y0:.3g}</text>',
        f'<text x="56" y="34" text-anchor="end">{y1:.3g}</text>',
        "</svg>",
    ]
    return "\n".join(out) + "\n"
