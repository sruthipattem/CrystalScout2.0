"""Server-rendered SVG charts (no JS chart library, works offline).

Colour roles come from CSS custom properties defined in static/style.css so
light/dark themes swap in one place. Series slots follow the validated
categorical order (blue, orange, aqua); the published Sendek equation is a
neutral dashed reference, never a fourth hue.
"""
from __future__ import annotations

import html
import math
from typing import Dict, List, Optional, Sequence, Tuple

SERIES_VAR = {"logreg": "var(--s1)", "random_forest": "var(--s2)", "gboost": "var(--s3)", "sendek2019": "var(--ref)"}
SERIES_DASH = {"sendek2019": "5 4"}


def _e(s) -> str:
    return html.escape(str(s), quote=True)


def roc_chart(curves: Dict[str, Dict], labels: Dict[str, str], aucs: Dict[str, float], title: str,
              w: int = 360, h: int = 320) -> str:
    pad_l, pad_b, pad_t, pad_r = 44, 40, 14, 12
    iw, ih = w - pad_l - pad_r, h - pad_t - pad_b

    def X(v):
        return pad_l + v * iw

    def Y(v):
        return pad_t + (1 - v) * ih

    parts = [f'<svg class="chart" viewBox="0 0 {w} {h}" role="img" aria-label="{_e(title)}">']
    for t in (0, .25, .5, .75, 1):
        parts.append(f'<line class="grid" x1="{X(0)}" x2="{X(1)}" y1="{Y(t)}" y2="{Y(t)}"/>')
        parts.append(f'<text class="tick" x="{X(0) - 6}" y="{Y(t) + 4}" text-anchor="end">{t:g}</text>')
        parts.append(f'<text class="tick" x="{X(t)}" y="{Y(0) + 16}" text-anchor="middle">{t:g}</text>')
    parts.append(f'<line class="diag" x1="{X(0)}" y1="{Y(0)}" x2="{X(1)}" y2="{Y(1)}"/>')
    parts.append(f'<text class="axis" x="{X(.5)}" y="{h - 6}" text-anchor="middle">False-positive rate</text>')
    parts.append(f'<text class="axis" transform="translate(12 {Y(.5)}) rotate(-90)" text-anchor="middle">True-positive rate</text>')
    for key, c in curves.items():
        pts = " ".join(f"{X(a):.1f},{Y(b):.1f}" for a, b in zip(c["fpr"], c["tpr"]))
        dash = f' stroke-dasharray="{SERIES_DASH[key]}"' if key in SERIES_DASH else ""
        parts.append(f'<polyline class="series" points="{pts}" style="stroke:{SERIES_VAR[key]}"{dash}>'
                     f'<title>{_e(labels[key])}: AUC {aucs[key]:.2f}</title></polyline>')
    parts.append("</svg>")
    legend = "".join(
        f'<li><span class="key{" dashed" if k in SERIES_DASH else ""}" style="--c:{SERIES_VAR[k]}"></span>'
        f'{_e(labels[k])} <b class="num">{aucs[k]:.2f}</b></li>' for k in curves)
    return f'<figure class="fig">{"".join(parts)}<ul class="legend">{legend}</ul></figure>'


def funnel(steps: List[Tuple[str, int]], highlight_last: bool = True) -> str:
    """Log-scaled sieve: each stage's bar length ~ log10(count)."""
    mx = math.log10(max(s[1] for s in steps) + 1)
    rows = []
    for i, (label, n) in enumerate(steps):
        frac = max(0.04, math.log10(n + 1) / mx)
        last = highlight_last and i == len(steps) - 1
        rows.append(
            f'<div class="sieve-row{" is-last" if last else ""}">'
            f'<span class="sieve-label">{_e(label)}</span>'
            f'<span class="sieve-track"><span class="sieve-bar" style="width:{frac * 100:.1f}%"></span></span>'
            f'<span class="sieve-n num">{n:,}</span></div>')
    return '<div class="sieve" role="list">' + "".join(rows) + "</div>"


def contributions(terms: List[Dict], w: int = 520) -> str:
    """Diverging bars of log-odds contributions around 0."""
    row_h, pad_l, pad_r = 30, 120, 20
    h = row_h * len(terms) + 26
    mx = max(0.5, max(abs(t["contribution"]) for t in terms))
    cx = pad_l + (w - pad_l - pad_r) / 2
    half = (w - pad_l - pad_r) / 2
    room = 46  # space reserved for value labels at the ends
    parts = [f'<svg class="chart" viewBox="0 0 {w} {h}" role="img" aria-label="Per-descriptor contribution to the score">']
    parts.append(f'<line class="zero" x1="{cx}" x2="{cx}" y1="4" y2="{h - 22}"/>')
    parts.append(f'<text class="tick" x="{cx - half}" y="{h - 6}">lowers P</text>')
    parts.append(f'<text class="tick" x="{cx + half}" y="{h - 6}" text-anchor="end">raises P</text>')
    for i, t in enumerate(terms):
        y = 6 + i * row_h
        v = t["contribution"]
        bw = abs(v) / mx * (half - room)
        x = cx if v >= 0 else cx - bw
        cls = "pos" if v >= 0 else "neg"
        parts.append(f'<text class="lab" x="{pad_l - 10}" y="{y + 15}" text-anchor="end">{_e(t["feature"])} '
                     f'<tspan class="muted">{t["value"]:.2f}</tspan></text>')
        parts.append(f'<rect class="bar {cls}" x="{x:.1f}" y="{y + 3}" width="{max(bw, 1.5):.1f}" height="16" rx="3">'
                     f'<title>{_e(t["name"])}: {v:+.2f} log-odds (value {t["value"]:.3f}, training mean {t["train_mean"]:.3f})</title></rect>')
        tx = (x + bw + 6) if v >= 0 else (x - 6)
        anchor = "start" if v >= 0 else "end"
        parts.append(f'<text class="val" x="{tx:.1f}" y="{y + 15}" text-anchor="{anchor}">{v:+.2f}</text>')
    parts.append("</svg>")
    return "".join(parts)


def strip(feature: str, pos: Sequence[float], neg: Sequence[float], value: float, lo: float, hi: float,
          w: int = 300) -> str:
    """Training distribution (superionic vs not) with the candidate marked."""
    h = 54
    pad = 8

    def X(v):
        v = min(max(v, lo), hi)
        return pad + (v - lo) / (hi - lo + 1e-12) * (w - 2 * pad)

    import random

    rnd = random.Random(7)
    parts = [f'<svg class="chart strip" viewBox="0 0 {w} {h}" role="img" aria-label="{_e(feature)} versus training data">']
    for v in neg:
        parts.append(f'<circle class="dot neg-dot" cx="{X(v):.1f}" cy="{30 + rnd.uniform(-5, 5):.1f}" r="2.2"/>')
    for v in pos:
        parts.append(f'<circle class="dot pos-dot" cx="{X(v):.1f}" cy="{14 + rnd.uniform(-5, 5):.1f}" r="2.2"/>')
    parts.append(f'<line class="marker" x1="{X(value):.1f}" x2="{X(value):.1f}" y1="2" y2="42"/>')
    xm = X(value)
    if xm - pad > 34:
        parts.append(f'<text class="tick" x="{pad}" y="{h - 1}">{lo:.2g}</text>')
    if (w - pad) - xm > 34:
        parts.append(f'<text class="tick" x="{w - pad}" y="{h - 1}" text-anchor="end">{hi:.2g}</text>')
    out = value < lo or value > hi
    anchor = "start" if xm < pad + 20 else ("end" if xm > w - pad - 20 else "middle")
    parts.append(f'<text class="tick strong" x="{xm:.1f}" y="{h - 1}" text-anchor="{anchor}">'
                 f'{value:.2f}{" (off scale)" if out else ""}</text>')
    parts.append("</svg>")
    return "".join(parts)


def hbars(items: List[Tuple[str, float]], fmt: str = "{:.0f}", w: int = 420, var: str = "var(--s1)",
          label_w: int = 150) -> str:
    row_h = 24
    h = row_h * len(items) + 6
    mx = max((v for _, v in items), default=1) or 1
    parts = [f'<svg class="chart" viewBox="0 0 {w} {h}" role="img">']
    for i, (lab, v) in enumerate(items):
        y = 3 + i * row_h
        bw = v / mx * (w - label_w - 56)
        parts.append(f'<text class="lab" x="{label_w - 8}" y="{y + 14}" text-anchor="end">{_e(lab)}</text>')
        parts.append(f'<rect class="bar" x="{label_w}" y="{y + 3}" width="{max(bw, 1.5):.1f}" height="14" rx="3" style="fill:{var}">'
                     f'<title>{_e(lab)}: {fmt.format(v)}</title></rect>')
        parts.append(f'<text class="val" x="{label_w + bw + 6:.1f}" y="{y + 14}">{fmt.format(v)}</text>')
    parts.append("</svg>")
    return "".join(parts)


def histogram(values: Sequence[float], bins: int, lo: float, hi: float, threshold: Optional[float] = None,
              xlabel: str = "", w: int = 560, h: int = 200) -> str:
    counts = [0] * bins
    for v in values:
        k = int((v - lo) / (hi - lo) * bins)
        counts[min(max(k, 0), bins - 1)] += 1
    mx = max(counts) or 1
    pad_l, pad_b, pad_t = 34, 34, 10
    iw, ih = w - pad_l - 10, h - pad_b - pad_t
    bw = iw / bins
    parts = [f'<svg class="chart" viewBox="0 0 {w} {h}" role="img" aria-label="{_e(xlabel)} distribution">']
    for t in range(0, mx + 1, max(1, mx // 4)):
        y = pad_t + ih - t / mx * ih
        parts.append(f'<line class="grid" x1="{pad_l}" x2="{w - 10}" y1="{y:.1f}" y2="{y:.1f}"/>'
                     f'<text class="tick" x="{pad_l - 6}" y="{y + 4:.1f}" text-anchor="end">{t}</text>')
    for i, c in enumerate(counts):
        x0 = lo + i * (hi - lo) / bins
        bh = c / mx * ih
        sup = threshold is not None and x0 >= threshold - 1e-9
        parts.append(f'<rect class="bar{" pos" if sup else " base"}" x="{pad_l + i * bw + 1:.1f}" y="{pad_t + ih - bh:.1f}" '
                     f'width="{bw - 2:.1f}" height="{bh:.1f}" rx="2"><title>{x0:.1f} to {x0 + (hi - lo) / bins:.1f}: {c}</title></rect>')
    for t in range(int(math.ceil(lo)), int(hi) + 1, 2):
        x = pad_l + (t - lo) / (hi - lo) * iw
        parts.append(f'<text class="tick" x="{x:.1f}" y="{pad_t + ih + 15}" text-anchor="middle">{t}</text>')
    if threshold is not None:
        x = pad_l + (threshold - lo) / (hi - lo) * iw
        parts.append(f'<line class="thresh" x1="{x:.1f}" x2="{x:.1f}" y1="{pad_t}" y2="{pad_t + ih}"/>'
                     f'<text class="tick strong" x="{x + 5:.1f}" y="{pad_t + 10}">superionic ≥ 10⁻⁴ S/cm</text>')
    parts.append(f'<text class="axis" x="{pad_l + iw / 2}" y="{h - 4}" text-anchor="middle">{_e(xlabel)}</text>')
    parts.append("</svg>")
    return "".join(parts)


def coef_compare(ours: Dict[str, float], theirs: Dict[str, float], w: int = 520) -> str:
    """Side-by-side signs and sizes of the two equations' coefficients."""
    feats = list(ours)
    row_h = 34
    h = row_h * len(feats) + 30
    mx = max(max(abs(v) for v in ours.values()), max(abs(theirs[k]) for k in feats))
    pad_l = 70
    half = (w - pad_l - 20) / 2
    cx = pad_l + half
    parts = [f'<svg class="chart" viewBox="0 0 {w} {h}" role="img" aria-label="Coefficient comparison">']
    parts.append(f'<line class="zero" x1="{cx}" x2="{cx}" y1="2" y2="{h - 24}"/>')
    parts.append(f'<text class="tick" x="{cx - half}" y="{h - 6}">negative</text><text class="tick" x="{cx + half}" y="{h - 6}" text-anchor="end">positive</text>')
    for i, k in enumerate(feats):
        y = 4 + i * row_h
        parts.append(f'<text class="lab" x="{pad_l - 10}" y="{y + 17}" text-anchor="end">{_e(k)}</text>')
        for j, (src, cls) in enumerate(((theirs, "ref-bar"), (ours, "our-bar"))):
            v = src[k]
            bw = abs(v) / mx * half
            x = cx if v >= 0 else cx - bw
            parts.append(f'<rect class="bar {cls}" x="{x:.1f}" y="{y + 2 + j * 13}" width="{max(bw, 1.5):.1f}" height="10" rx="2">'
                         f'<title>{"CrystalScout" if j else "Sendek 2019"} {k}: {v:+.2f}</title></rect>')
        same = (ours[k] > 0) == (theirs[k] > 0)
        parts.append(f'<text class="val" x="{w - 4}" y="{y + 17}" text-anchor="end">{"same sign" if same else "flipped"}</text>')
    parts.append("</svg>")
    return "".join(parts)


def parity(points: List[Dict], w: int = 520, h: int = 400, lo: float = -14, hi: float = 0) -> str:
    """Predicted vs measured log10 σ for held-out materials, with ±1 log band."""
    pad_l, pad_b, pad_t, pad_r = 46, 38, 10, 12
    X = lambda v: pad_l + (min(max(v, lo), hi) - lo) / (hi - lo) * (w - pad_l - pad_r)  # noqa: E731
    Y = lambda v: h - pad_b - (min(max(v, lo), hi) - lo) / (hi - lo) * (h - pad_b - pad_t)  # noqa: E731
    p = [f'<svg class="chart" viewBox="0 0 {w} {h}" role="img" aria-label="Predicted against measured conductivity">']
    for t in range(int(lo), int(hi) + 1, 2):
        p.append(f'<line class="grid" x1="{X(lo)}" x2="{X(hi)}" y1="{Y(t)}" y2="{Y(t)}"/>')
        p.append(f'<line class="grid" y1="{Y(lo)}" y2="{Y(hi)}" x1="{X(t)}" x2="{X(t)}"/>')
        p.append(f'<text class="tick" x="{X(lo) - 6}" y="{Y(t) + 4}" text-anchor="end">{t}</text>')
        p.append(f'<text class="tick" x="{X(t)}" y="{Y(lo) + 16}" text-anchor="middle">{t}</text>')
    band = (f"{X(lo)},{Y(lo + 1)} {X(hi - 1)},{Y(hi)} {X(hi)},{Y(hi)} {X(hi)},{Y(hi - 1)} {X(lo + 1)},{Y(lo)} {X(lo)},{Y(lo)}")
    p.append(f'<polygon class="band" points="{band}"/>')
    p.append(f'<line class="diag" x1="{X(lo)}" y1="{Y(lo)}" x2="{X(hi)}" y2="{Y(hi)}"/>')
    p.append(f'<line class="thresh" x1="{X(-4)}" x2="{X(-4)}" y1="{Y(lo)}" y2="{Y(hi)}"/>')
    p.append(f'<line class="thresh" y1="{Y(-4)}" y2="{Y(-4)}" x1="{X(lo)}" x2="{X(hi)}"/>')
    for q in points:
        p.append(f'<circle class="dot pt" cx="{X(q["true"]):.1f}" cy="{Y(q["pred"]):.1f}" r="3.6">'
                 f'<title>{_e(q["formula"])}: measured {q["true"]:.1f}, predicted {q["pred"]:.1f}</title></circle>')
    p.append(f'<text class="axis" x="{(X(lo) + X(hi)) / 2}" y="{h - 4}" text-anchor="middle">Measured log₁₀ σ (S/cm)</text>')
    p.append(f'<text class="axis" transform="translate(12 {(Y(lo) + Y(hi)) / 2}) rotate(-90)" text-anchor="middle">Predicted log₁₀ σ</text>')
    p.append("</svg>")
    return "".join(p)


def sigma_intervals(items: List[Dict], w: int = 620, lo: float = -12, hi: float = -1) -> str:
    """One row per material: predicted log10 σ with its 80 % interval."""
    row_h, label_w, top = 34, 150, 22
    h = top + row_h * len(items) + 26
    X = lambda v: label_w + (min(max(v, lo), hi) - lo) / (hi - lo) * (w - label_w - 16)  # noqa: E731
    colors = ["#6246ea", "#14b8a6", "#f07a3e", "#2563eb"]
    p = [f'<svg class="chart" viewBox="0 0 {w} {h}" role="img" aria-label="Predicted conductivity ranges">']
    for t in range(int(lo), int(hi) + 1):
        p.append(f'<line class="grid" x1="{X(t)}" x2="{X(t)}" y1="{top - 6}" y2="{h - 22}"/>')
        if t % 2 == 0 or t == -1:
            p.append(f'<text class="tick" x="{X(t)}" y="{h - 8}" text-anchor="middle">10^{t}</text>')
    p.append(f'<line class="thresh" x1="{X(-4)}" x2="{X(-4)}" y1="{top - 10}" y2="{h - 22}"/>')
    p.append(f'<text class="tick strong" x="{X(-4) + 4}" y="{top - 10}">superionic</text>')
    p.append(f'<line class="thresh" x1="{X(-3)}" x2="{X(-3)}" y1="{top - 10}" y2="{h - 22}" style="stroke:var(--good)"/>')
    for i, it in enumerate(items):
        y = top + i * row_h + row_h / 2
        c = colors[i % len(colors)]
        p.append(f'<text class="lab" x="{label_w - 10}" y="{y + 4}" text-anchor="end">{_e(it["label"])}</text>')
        if it.get("mu") is None:
            continue
        p.append(f'<line class="ivline" x1="{X(it["lo"]):.1f}" x2="{X(it["hi"]):.1f}" y1="{y}" y2="{y}" style="stroke:{c}"/>')
        p.append(f'<circle class="ivdot" cx="{X(it["mu"]):.1f}" cy="{y}" r="7" style="fill:{c}"><title>{_e(it["label"])}: '
                 f'10^{it["mu"]:.1f} S/cm (80 % range 10^{it["lo"]:.1f} to 10^{it["hi"]:.1f})</title></circle>')
    p.append("</svg>")
    return "".join(p)
