"""Self-hosted image CAPTCHA (no third-party service, works offline).

Each challenge is five characters drawn as distorted vector strokes in an SVG
image, so the characters never appear as text in the page and must be read
from the picture. The answer is never stored in readable form: the session
keeps only an HMAC of it, plus the time it was issued. A challenge is valid
for one attempt and five minutes.
"""
from __future__ import annotations

import base64
import hashlib
import hmac
import math
import random
import secrets
import time
from typing import Dict, List, Tuple

ALPHABET = "ACDEFHKMNPRTUVWXYZ234679"
LENGTH = 5
TTL = 300  # seconds

# Stroke font on a 10 x 14 grid: each glyph is a list of polylines.
P = List[Tuple[float, float]]
GLYPHS: Dict[str, List[P]] = {
    "A": [[(0, 14), (5, 0), (10, 14)], [(2.2, 8), (7.8, 8)]],
    "C": [[(10, 2), (7, 0), (3, 0), (0, 3), (0, 11), (3, 14), (7, 14), (10, 12)]],
    "D": [[(0, 0), (0, 14), (6, 14), (10, 10), (10, 4), (6, 0), (0, 0)]],
    "E": [[(10, 0), (0, 0), (0, 14), (10, 14)], [(0, 7), (7, 7)]],
    "F": [[(10, 0), (0, 0), (0, 14)], [(0, 7), (7, 7)]],
    "H": [[(0, 0), (0, 14)], [(10, 0), (10, 14)], [(0, 7), (10, 7)]],
    "K": [[(0, 0), (0, 14)], [(10, 0), (0, 8)], [(3, 6), (10, 14)]],
    "M": [[(0, 14), (0, 0), (5, 8), (10, 0), (10, 14)]],
    "N": [[(0, 14), (0, 0), (10, 14), (10, 0)]],
    "P": [[(0, 14), (0, 0), (7, 0), (10, 2), (10, 5), (7, 7), (0, 7)]],
    "R": [[(0, 14), (0, 0), (7, 0), (10, 2), (10, 5), (7, 7), (0, 7)], [(5, 7), (10, 14)]],
    "T": [[(0, 0), (10, 0)], [(5, 0), (5, 14)]],
    "U": [[(0, 0), (0, 11), (3, 14), (7, 14), (10, 11), (10, 0)]],
    "V": [[(0, 0), (5, 14), (10, 0)]],
    "W": [[(0, 0), (2.5, 14), (5, 6), (7.5, 14), (10, 0)]],
    "X": [[(0, 0), (10, 14)], [(10, 0), (0, 14)]],
    "Y": [[(0, 0), (5, 7), (10, 0)], [(5, 7), (5, 14)]],
    "Z": [[(0, 0), (10, 0), (0, 14), (10, 14)]],
    "2": [[(0, 3), (3, 0), (7, 0), (10, 3), (10, 5), (0, 14), (10, 14)]],
    "3": [[(0, 1), (3, 0), (7, 0), (10, 2), (10, 5), (6, 7), (10, 9), (10, 12), (7, 14), (3, 14), (0, 13)],
          [(3, 7), (6, 7)]],
    "4": [[(8, 14), (8, 0), (0, 10), (10, 10)]],
    "6": [[(9, 1), (6, 0), (3, 0), (0, 4), (0, 11), (3, 14), (7, 14), (10, 11), (10, 9), (7, 6), (3, 6),
           (0, 9)]],
    "7": [[(0, 0), (10, 0), (4, 14)]],
    "9": [[(10, 5), (7, 8), (3, 8), (0, 5), (0, 3), (3, 0), (7, 0), (10, 3), (10, 10), (7, 14), (3, 14),
           (1, 13)]],
}
INK = ["#3b2a8f", "#1f4fa8", "#0f6e74", "#5b3fb0", "#273046", "#7a2e8e"]


def new_code(rng: random.Random = None) -> str:
    rng = rng or random.SystemRandom()
    return "".join(rng.choice(ALPHABET) for _ in range(LENGTH))


def render_svg(code: str, seed: int = None) -> str:
    """Draw ``code`` as a distorted SVG image (220 x 72)."""
    rng = random.Random(seed if seed is not None else secrets.randbits(32))
    w, h = 220, 72
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}">',
             '<defs><linearGradient id="g" x1="0" y1="0" x2="1" y2="1">'
             '<stop offset="0" stop-color="#f4f1ff"/><stop offset="1" stop-color="#e8f6f5"/></linearGradient></defs>',
             f'<rect width="{w}" height="{h}" rx="10" fill="url(#g)"/>']
    for _ in range(46):  # speckle
        parts.append(f'<circle cx="{rng.uniform(0, w):.1f}" cy="{rng.uniform(0, h):.1f}" r="{rng.uniform(.6, 1.8):.1f}" '
                     f'fill="{rng.choice(INK)}" opacity="{rng.uniform(.15, .45):.2f}"/>')
    for _ in range(5):  # interfering curves
        y0, y1 = rng.uniform(8, h - 8), rng.uniform(8, h - 8)
        parts.append(f'<path d="M-5 {y0:.1f} C {rng.uniform(40, 90):.1f} {rng.uniform(0, h):.1f}, '
                     f'{rng.uniform(120, 180):.1f} {rng.uniform(0, h):.1f}, {w + 5} {y1:.1f}" fill="none" '
                     f'stroke="{rng.choice(INK)}" stroke-width="{rng.uniform(1, 2.2):.1f}" opacity="{rng.uniform(.25, .5):.2f}"/>')
    amp, freq, phase = rng.uniform(2.5, 4.5), rng.uniform(.035, .06), rng.uniform(0, 6.3)
    x = 16.0
    for ch in code:
        scale = rng.uniform(2.6, 3.2)
        ang = math.radians(rng.uniform(-20, 20))
        skew = rng.uniform(-.25, .25)
        cy = h / 2 + rng.uniform(-5, 5)
        ca, sa = math.cos(ang), math.sin(ang)
        color = rng.choice(INK)
        sw = rng.uniform(2.4, 3.4)
        for line in GLYPHS[ch]:
            pts = []
            for gx, gy in line:
                gx += rng.uniform(-.45, .45)
                gy += rng.uniform(-.45, .45)
                lx, ly = (gx - 5 + skew * (gy - 7)) * scale, (gy - 7) * scale
                px = x + 14 + lx * ca - ly * sa
                py = cy + lx * sa + ly * ca
                py += amp * math.sin(freq * px + phase)
                pts.append(f"{px:.1f},{py:.1f}")
            parts.append(f'<polyline points="{" ".join(pts)}" fill="none" stroke="{color}" stroke-width="{sw:.1f}" '
                         f'stroke-linecap="round" stroke-linejoin="round"/>')
        x += rng.uniform(36, 41)
    parts.append(f'<path d="M0 {rng.uniform(20, 52):.1f} Q {w / 2:.0f} {rng.uniform(0, h):.1f} {w} {rng.uniform(20, 52):.1f}" '
                 f'fill="none" stroke="#fff" stroke-width="1.6" opacity=".7"/>')
    parts.append("</svg>")
    return "".join(parts)


def data_uri(svg: str) -> str:
    return "data:image/svg+xml;base64," + base64.b64encode(svg.encode()).decode()


def digest(secret: str, code: str, nonce: str) -> str:
    return hmac.new(secret.encode(), f"{nonce}:{code.strip().upper()}".encode(), hashlib.sha256).hexdigest()


def issue(session, secret: str) -> str:
    """Create a challenge, remember only its HMAC in the session, return the image data-URI."""
    code = new_code()
    nonce = secrets.token_hex(8)
    session["cap"] = {"h": digest(secret, code, nonce), "n": nonce, "t": int(time.time())}
    return data_uri(render_svg(code))


def check(session, secret: str, answer: str) -> Tuple[bool, str]:
    """Verify and burn the current challenge (one attempt per image)."""
    cap = session.pop("cap", None)
    if not cap:
        return False, "The security check expired. A new one is shown; please try again."
    if time.time() - cap.get("t", 0) > TTL:
        return False, "The security check expired after 5 minutes. Please type the new characters."
    if not answer or not hmac.compare_digest(cap["h"], digest(secret, answer, cap["n"])):
        return False, "The security characters didn't match. Please type the new ones shown."
    return True, ""
