"""Sign-in for the CrystalScout platform: four role-based accounts + image CAPTCHA.

Account passwords must be set with CRYSTALSCOUT_PW_<USERNAME> environment
variables, e.g. CRYSTALSCOUT_PW_ADMIN.

    username    role         access
    admin       admin        everything + Admin console (accounts, sign-in audit log)
    scientist   scientist    full screening, scoring, live Materials Project, exports
    partner     partner      same as scientist (industry battery R&D partner)
    expo        guest        everything except live Materials Project queries and the Admin console

Security
* Passwords are held only as salted hashes (werkzeug), checked in constant time.
* A self-hosted image CAPTCHA (webapp/captcha.py) is required on every sign-in.
* 5 wrong passwords for one username within 10 minutes lock that username for
  10 minutes; 25 failures from one address lock that address.
* Forms carry a per-session CSRF token; cookies are HttpOnly + SameSite=Lax.
* Every sign-in attempt is written to data/access_log.jsonl and shown to admins.
* Changing an account's password signs that account out everywhere.
"""
from __future__ import annotations

import base64
import hashlib
import json
import os
import secrets
import threading
import time
from collections import deque
from functools import wraps
from typing import Dict, List, Optional
from urllib.parse import urlparse

from flask import (Blueprint, abort, current_app, flash, g, has_request_context, jsonify, make_response, redirect,
                   render_template, request, session, url_for)
from werkzeug.security import check_password_hash, generate_password_hash

from crystalscout.config import DATA_DIR

from . import captcha

ROLES = {
    "admin": {"label": "Administrator", "color": "violet"},
    "scientist": {"label": "Research Scientist", "color": "blue"},
    "partner": {"label": "Industry Partner", "color": "teal"},
    "guest": {"label": "Expo Guest", "color": "amber"},
}

ACCOUNTS: Dict[str, Dict] = {
    "admin": {"name": "Platform Administrator", "role": "admin", "org": "CrystalScout · Group G1544"},
    "scientist": {"name": "Research Scientist", "role": "scientist", "org": "Materials Discovery Lab"},
    "partner": {"name": "Battery R&D Partner", "role": "partner", "org": "Solid-State Cell Development"},
    "expo": {"name": "Expo Visitor", "role": "guest", "org": "Project Expo 2026"},
}

LOCK_ATTEMPTS = 5
LOCK_WINDOW = 600  # seconds
IP_LOCK_ATTEMPTS = 25
PUBLIC_ENDPOINTS = {"auth.login", "auth.register", "auth.captcha_image", "static"}
LOG_FILE = DATA_DIR / "access_log.jsonl"

bp = Blueprint("auth", __name__)
_failures: Dict[str, List[float]] = {}
_ip_failures: Dict[str, List[float]] = {}
_last_login: Dict[str, str] = {}
_log_lock = threading.Lock()
_recent: deque = deque(maxlen=500)


def _now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def _build_users() -> Dict[str, Dict]:
    users = {}
    missing = []
    for uname, a in ACCOUNTS.items():
        env_name = f"CRYSTALSCOUT_PW_{uname.upper()}"
        pw = os.environ.get(env_name)
        if not pw:
            missing.append(env_name)
            continue
        users[uname] = {
            "username": uname, "full_name": a["name"], "role": a["role"], "org": a.get("org", ""),
            "hash": generate_password_hash(pw),
            "ver": hashlib.sha256(f"{uname}\0{pw}".encode()).hexdigest()[:16],
        }
    if missing:
        raise RuntimeError("Set account password environment variables before starting: " + ", ".join(missing))
    return users


USERS = _build_users()


def account_for(username: str) -> Optional[Dict]:
    return USERS.get((username or "").strip().lower())


def check_credentials(username: str, password: str) -> Optional[Dict]:
    """Return the account when username + password match, else None (constant time)."""
    u = account_for(username)
    ok = check_password_hash(u["hash"] if u else USERS["admin"]["hash"], password or "")
    return u if (u and ok) else None


def public_user(u: Dict) -> Dict:
    """What templates and the API see (never the hash)."""
    return {"id": u["username"], "username": u["username"], "full_name": u["full_name"], "role": u["role"],
            "role_label": ROLES[u["role"]]["label"], "role_color": ROLES[u["role"]]["color"], "org": u["org"],
            "is_admin": 1 if u["role"] == "admin" else 0, "can_live": u["role"] != "guest",
            "initials": "".join(w[0] for w in u["full_name"].split()[:2]).upper(),
            "last_login": _last_login.get(u["username"])}


# ------------------------------------------------------------------ audit log
def log_event(event: str, username: str, ok: bool, detail: str = "", ip: str = "") -> None:
    ctx = has_request_context()
    rec = {"time": _now(), "event": event, "username": (username or "")[:40], "ok": bool(ok), "detail": detail,
           "ip": ip or (request.remote_addr if ctx else "") or "",
           "agent": request.headers.get("User-Agent", "")[:120] if ctx else "api"}
    _recent.appendleft(rec)
    try:
        if ctx and current_app.config.get("TESTING"):
            return
    except RuntimeError:
        pass
    if os.environ.get("CRYSTALSCOUT_NO_AUDIT"):
        return
    try:
        with _log_lock, open(LOG_FILE, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(rec) + "\n")
    except OSError:
        pass


def recent_events(n: int = 200) -> List[Dict]:
    if LOG_FILE.exists():
        try:
            lines = LOG_FILE.read_text(encoding="utf-8").splitlines()[-n:]
            disk = [json.loads(x) for x in reversed(lines) if x.strip()]
            if disk:
                return disk
        except (OSError, ValueError):
            pass
    return list(_recent)[:n]


# ------------------------------------------------------------------ helpers
def _prune(store: Dict[str, List[float]], key: str) -> List[float]:
    now = time.time()
    hits = [t for t in store.get(key, []) if now - t < LOCK_WINDOW]
    store[key] = hits
    return hits


def _locked(username: str = "", ip: Optional[str] = None) -> int:
    """Seconds left on a lockout for this username or address (0 = not locked)."""
    now = time.time()
    hits = _prune(_failures, (username or "").lower())
    if len(hits) >= LOCK_ATTEMPTS:
        return int(LOCK_WINDOW - (now - hits[0])) + 1
    if ip is None and has_request_context():
        ip = request.remote_addr
    ih = _prune(_ip_failures, ip or "?")
    if len(ih) >= IP_LOCK_ATTEMPTS:
        return int(LOCK_WINDOW - (now - ih[0])) + 1
    return 0


def _fail(username: str, ip: Optional[str] = None) -> None:
    if ip is None and has_request_context():
        ip = request.remote_addr
    _failures.setdefault((username or "").lower(), []).append(time.time())
    _ip_failures.setdefault(ip or "?", []).append(time.time())


def attempts_left(username: str) -> int:
    return LOCK_ATTEMPTS - len(_prune(_failures, (username or "").lower()))


def _safe_next(target: Optional[str]) -> str:
    if target:
        p = urlparse(target)
        if (not p.scheme and not p.netloc and target.startswith("/") and not target.startswith("//")
                and not target.startswith("/login") and not target.startswith("/captcha")):
            return target
    return url_for("overview")


def csrf_token() -> str:
    if "_csrf" not in session:
        session["_csrf"] = secrets.token_urlsafe(32)
    return session["_csrf"]


def check_csrf() -> None:
    if current_app.config.get("TESTING") and not current_app.config.get("CSRF_IN_TESTS"):
        return
    sent = request.form.get("_csrf") or request.headers.get("X-CSRF-Token", "")
    if not sent or not secrets.compare_digest(sent, session.get("_csrf", "")):
        abort(400, description="Your session expired. Reload the page and try again.")


def _captcha_required() -> bool:
    return not current_app.config.get("TESTING") or current_app.config.get("CAPTCHA_IN_TESTS", False)


def current_user() -> Optional[Dict]:
    return getattr(g, "user", None)


def admin_required(fn):
    @wraps(fn)
    def wrapper(*a, **kw):
        u = current_user()
        if not u or not u["is_admin"]:
            abort(403)
        return fn(*a, **kw)
    return wrapper


# ------------------------------------------------------------------- wiring
def init_auth(app) -> None:
    app.register_blueprint(bp)
    app.config.setdefault("SESSION_COOKIE_HTTPONLY", True)
    app.config.setdefault("SESSION_COOKIE_SAMESITE", "Lax")
    app.config.setdefault("PERMANENT_SESSION_LIFETIME", 60 * 60 * 12)
    app.jinja_env.globals["csrf_token"] = csrf_token
    app.jinja_env.globals["ROLES"] = ROLES

    @app.before_request
    def _load_user_and_guard():
        u = account_for(session.get("user", ""))
        signed_in = bool(u) and session.get("v") == u["ver"]
        if session.get("user") and not signed_in:  # password changed since this session started
            session.clear()
        g.user = public_user(u) if signed_in else None
        if request.endpoint in PUBLIC_ENDPOINTS or request.endpoint is None:
            return None
        if g.user is None:
            if request.path.startswith("/api/"):
                return jsonify({"error": "sign in required"}), 401
            return redirect(url_for("auth.login", next=request.full_path.rstrip("?")))
        if request.method == "POST" and not request.path.startswith("/api/"):
            check_csrf()
        return None

    @app.after_request
    def _security_headers(resp):
        resp.headers.setdefault("X-Content-Type-Options", "nosniff")
        resp.headers.setdefault("X-Frame-Options", "SAMEORIGIN")
        resp.headers.setdefault("Referrer-Policy", "same-origin")
        return resp

    @app.context_processor
    def _inject_user():
        return {"current_user": current_user()}


# ------------------------------------------------------------------- routes
@bp.route("/captcha.svg")
def captcha_image():
    """A fresh challenge image (the refresh button on the sign-in page)."""
    uri = captcha.issue(session, current_app.secret_key)
    resp = make_response(base64.b64decode(uri.split(",", 1)[1]))
    resp.headers["Content-Type"] = "image/svg+xml"
    resp.headers["Cache-Control"] = "no-store, max-age=0"
    return resp


@bp.route("/login", methods=["GET", "POST"])
def login():
    if current_user() is not None:
        return redirect(url_for("overview"))
    error = None
    username = ""
    field = None
    if request.method == "POST":
        check_csrf()
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "")
        wait = _locked(username) if username else 0
        ok_captcha, cap_msg = True, ""
        if _captcha_required():
            ok_captcha, cap_msg = captcha.check(session, current_app.secret_key, request.form.get("captcha", ""))
        if wait:
            error = f"Too many wrong attempts. Sign-in for this account is paused for {max(1, wait // 60)} minute(s)."
            log_event("login", username, False, "locked out")
        elif not username or not password:
            error, field = "Enter your username and password.", "username"
        elif not ok_captcha:
            error, field = cap_msg, "captcha"
            log_event("login", username, False, "captcha failed")
        else:
            u = check_credentials(username, password)
            if u is None:
                _fail(username)
                left = attempts_left(username)
                error = "That username and password don't match."
                if 0 < left <= 3:
                    error += f" {left} attempt{'s' if left != 1 else ''} left before a 10-minute pause."
                field = "password"
                log_event("login", username, False, "wrong credentials")
            else:
                _failures.pop(u["username"], None)
                nxt = request.args.get("next")
                session.clear()
                session.permanent = request.form.get("remember") == "on"
                session["user"] = u["username"]
                session["v"] = u["ver"]
                csrf_token()
                _last_login[u["username"]] = _now()
                log_event("login", u["username"], True, "signed in")
                flash(f"Welcome, {u['full_name']}. You're signed in as {ROLES[u['role']]['label']}.", "ok")
                return redirect(_safe_next(nxt))
    cap = captcha.issue(session, current_app.secret_key)
    return (render_template("auth/login.html", error=error, username=username, captcha_uri=cap, field=field),
            401 if error else 200)


@bp.route("/register", methods=["GET", "POST"])
def register():
    """Self-registration is off: accounts are issued by the administrator."""
    flash("Accounts are issued by the CrystalScout administrator. Sign in with the details you were given.", "warn")
    return redirect(url_for("auth.login"))


@bp.route("/logout", methods=["POST"])
def logout():
    check_csrf()
    u = current_user()
    if u:
        log_event("logout", u["username"], True, "signed out")
    session.clear()
    flash("You're signed out. See you soon.", "ok")
    return redirect(url_for("auth.login"))


@bp.route("/account", methods=["GET"])
def account():
    return render_template("auth/account.html")


@bp.route("/admin")
@admin_required
def admin():
    users = []
    for u in USERS.values():
        pu = public_user(u)
        pu["locked"] = _locked(u["username"], ip="-") > 0
        users.append(pu)
    events = recent_events(200)
    stats = {"ok": sum(1 for e in events if e["ok"] and e["event"] == "login"),
             "failed": sum(1 for e in events if not e["ok"]),
             "captcha": sum(1 for e in events if e.get("detail") == "captcha failed"),
             "locked": sum(1 for e in events if e.get("detail") == "locked out")}
    return render_template("auth/admin.html", users=users, events=events[:100], stats=stats)
