"""API sign-in: the SAME four accounts as the web app (webapp/auth.py).

The API has no CAPTCHA (it is for programs); the per-account lockout still applies.

POST /auth/login with the username and password returns a signed token.
Send it on every other request as:   Authorization: Bearer <token>
Tokens expire after CRYSTALSCOUT_API_TOKEN_HOURS (default 12) and stop
working if the username or password is changed.
"""
from __future__ import annotations

import os
import secrets
from typing import Optional

from itsdangerous import BadSignature, SignatureExpired, URLSafeTimedSerializer

from crystalscout.config import DATA_DIR
from webapp import auth as web_auth

TOKEN_HOURS = float(os.environ.get("CRYSTALSCOUT_API_TOKEN_HOURS", "12"))


def _secret() -> str:
    env = os.environ.get("CRYSTALSCOUT_SECRET")
    if env:
        return env
    f = DATA_DIR / ".secret_key"  # shared with the Flask app
    if not f.exists():
        f.write_text(secrets.token_hex(32))
    return f.read_text().strip()


_signer = URLSafeTimedSerializer(_secret(), salt="crystalscout-api")


class LoginError(Exception):
    """Wrong credentials or locked out (becomes HTTP 401 / 429)."""

    def __init__(self, message: str, locked: bool = False):
        super().__init__(message)
        self.locked = locked


def login(username: str, password: str) -> dict:
    username = (username or "").strip()
    wait = web_auth._locked(username, ip="api")
    if wait:
        raise LoginError(f"too many wrong attempts; try again in {max(1, wait // 60)} minute(s)", locked=True)
    u = web_auth.check_credentials(username, password)
    if u is None:
        web_auth._fail(username, ip="api")
        web_auth.log_event("api-login", username, False, "wrong credentials", ip="api")
        raise LoginError("wrong username or password")
    web_auth._failures.pop(u["username"], None)
    web_auth.log_event("api-login", u["username"], True, "token issued", ip="api")
    token = _signer.dumps({"u": u["username"], "v": u["ver"]})
    return {"access_token": token, "token_type": "bearer", "expires_in": int(TOKEN_HOURS * 3600),
            "username": u["username"], "role": u["role"]}


def verify(token: Optional[str]) -> str:
    """Return the username for a valid token, else raise LoginError."""
    if not token:
        raise LoginError("sign in first: POST /auth/login, then send 'Authorization: Bearer <token>'")
    try:
        data = _signer.loads(token, max_age=int(TOKEN_HOURS * 3600))
    except SignatureExpired:
        raise LoginError("token expired; sign in again")
    except BadSignature:
        raise LoginError("invalid token")
    u = web_auth.account_for(data.get("u", ""))
    if not u or data.get("v") != u["ver"]:
        raise LoginError("token no longer valid; sign in again")
    return u["username"]


def user_info(username: str) -> dict:
    u = web_auth.account_for(username)
    return web_auth.public_user(u) if u else {}
