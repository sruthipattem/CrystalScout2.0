"""Four role-based accounts, image CAPTCHA, lockout, access control and CSRF."""
import io
import re

import pytest

from crystalscout.config import OBELIX_DIR

ACCOUNTS = {
    "admin": "test-only-admin-password",
    "scientist": "test-only-scientist-password",
    "partner": "test-only-partner-password",
    "expo": "test-only-expo-password",
}
USER, PASSWORD = "scientist", ACCOUNTS["scientist"]


@pytest.fixture()
def app():
    from webapp import auth
    from webapp.app import app as flask_app

    flask_app.config["TESTING"] = True
    flask_app.config["CSRF_IN_TESTS"] = False
    flask_app.config["CAPTCHA_IN_TESTS"] = False
    auth._failures.clear()
    auth._ip_failures.clear()
    return flask_app


def login(c, u=USER, p=PASSWORD, **extra):
    return c.post("/login", data={"username": u, "password": p, **extra})


def token(client, url="/login"):
    html = client.get(url).data.decode()
    return re.search(r'name="_csrf" value="([^"]+)"', html).group(1)


def test_pages_require_sign_in(app):
    c = app.test_client()
    r = c.get("/shortlist")
    assert r.status_code == 302 and "/login" in r.location and "next=" in r.location
    assert c.post("/api/predict", json={}).status_code == 401
    r = c.get("/login")
    assert r.status_code == 200 and b"Security check" in r.data and b"data:image/svg+xml;base64," in r.data


@pytest.mark.parametrize("user", list(ACCOUNTS))
def test_each_of_the_four_accounts_signs_in_to_the_dashboard(app, user):
    c = app.test_client()
    r = login(c, user, ACCOUNTS[user])
    assert r.status_code == 302 and r.location.endswith("/")
    page = c.get("/")
    assert page.status_code == 200 and b"Top candidates right now" in page.data


def test_wrong_credentials_rejected(app):
    c = app.test_client()
    assert login(c, USER, "wrong-password").status_code == 401
    assert login(c, "admin", PASSWORD).status_code == 401            # another account's password
    assert login(c, "nobody", PASSWORD).status_code == 401
    assert login(c, "", "").status_code == 401
    assert login(c, USER.upper(), PASSWORD).status_code == 302       # usernames are case-insensitive


def test_captcha_enforced(app):
    from webapp import captcha

    app.config["CAPTCHA_IN_TESTS"] = True
    try:
        c = app.test_client()
        c.get("/login")
        r = login(c, captcha="WRONG")
        assert r.status_code == 401 and b"security characters" in r.data
        # the answer is never readable from the session: only an HMAC is kept
        with c.session_transaction() as s:
            cap = s["cap"]
            assert set(cap) == {"h", "n", "t"}
        # forge a known challenge, then answer it
        with c.session_transaction() as s:
            s["cap"] = {"h": captcha.digest(app.secret_key, "AC234", "n1"), "n": "n1", "t": int(__import__("time").time())}
        assert login(c, captcha="ac234").status_code == 302       # case-insensitive
    finally:
        app.config["CAPTCHA_IN_TESTS"] = False


def test_captcha_one_attempt_and_refresh(app):
    from webapp import captcha

    app.config["CAPTCHA_IN_TESTS"] = True
    try:
        c = app.test_client()
        r = c.get("/captcha.svg")
        assert r.status_code == 200 and r.mimetype == "image/svg+xml" and b"<polyline" in r.data
        with c.session_transaction() as s:
            s["cap"] = {"h": captcha.digest(app.secret_key, "XYZ34", "n"), "n": "n", "t": int(__import__("time").time())}
        assert login(c, p="bad", captcha="XYZ34").status_code == 401   # wrong password burns the image
        assert login(c, captcha="XYZ34").status_code == 401            # same answer no longer valid
    finally:
        app.config["CAPTCHA_IN_TESTS"] = False


def test_captcha_render_is_distorted_svg():
    from webapp import captcha

    a, b = captcha.render_svg("AC234", seed=1), captcha.render_svg("AC234", seed=2)
    assert a != b and a.startswith("<svg") and "AC234" not in a


def test_registration_is_disabled(app):
    c = app.test_client()
    r = c.get("/register")
    assert r.status_code == 302 and "/login" in r.location
    assert login(c, "newuser", "longenough1").status_code == 401


def test_lockout_after_five_wrong_passwords(app):
    c = app.test_client()
    for _ in range(5):
        assert login(c, USER, "nope-nope").status_code == 401
    r = login(c)
    assert r.status_code == 401 and b"Too many wrong attempts" in r.data
    assert login(c, "partner", ACCOUNTS["partner"]).status_code == 302   # other accounts unaffected


def test_passwords_not_stored_in_plain_text():
    from webapp import auth

    for u, pw in ACCOUNTS.items():
        assert pw not in auth.USERS[u]["hash"]


def test_missing_account_password_prevents_startup(monkeypatch):
    from webapp import auth

    monkeypatch.delenv("CRYSTALSCOUT_PW_ADMIN")
    with pytest.raises(RuntimeError, match="CRYSTALSCOUT_PW_ADMIN"):
        auth._build_users()


def test_roles_and_admin_console(app):
    c = app.test_client()
    login(c, "scientist", ACCOUNTS["scientist"])
    assert c.get("/admin").status_code == 403
    c2 = app.test_client()
    login(c2, "admin", ACCOUNTS["admin"])
    r = c2.get("/admin")
    assert r.status_code == 200 and b"Sign-in audit log" in r.data
    for pw in ACCOUNTS.values():
        assert pw.encode() not in r.data


def test_expo_guest_cannot_run_live(app):
    c = app.test_client()
    login(c, "expo", ACCOUNTS["expo"])
    r = c.post("/screen", data={"source": "live", "band_gap_min": "1", "ehull_max": "0", "exclude_tm": "on"})
    assert r.status_code == 403


def test_logout_and_account_page(app):
    c = app.test_client()
    login(c)
    r = c.get("/account")
    assert r.status_code == 200 and b"Research Scientist" in r.data
    assert c.post("/logout").status_code == 302
    assert c.get("/").status_code == 302


def test_old_session_rejected_when_password_changes(app, monkeypatch):
    from webapp import auth

    c = app.test_client()
    login(c)
    assert c.get("/").status_code == 200
    monkeypatch.setitem(auth.USERS[USER], "ver", "changed")
    assert c.get("/").status_code == 302


def test_open_redirect_blocked(app):
    c = app.test_client()
    r = c.post("/login?next=https://evil.example/x", data={"username": USER, "password": PASSWORD})
    assert r.status_code == 302 and "evil" not in r.location
    c2 = app.test_client()
    r = c2.post("/login?next=/models", data={"username": USER, "password": PASSWORD})
    assert r.location.endswith("/models")


def test_csrf_enforced(app):
    app.config["CSRF_IN_TESTS"] = True
    try:
        c = app.test_client()
        assert login(c).status_code == 400
        t = token(c)
        assert login(c, _csrf=t).status_code == 302
        assert c.post("/logout").status_code == 400            # no token -> rejected, still signed in
        assert c.get("/").status_code == 200
        t = token(c, "/")
        assert c.post("/logout", data={"_csrf": t}).status_code == 302
    finally:
        app.config["CSRF_IN_TESTS"] = False


def test_quick_check(app):
    c = app.test_client()
    login(c)
    assert c.get("/simple").status_code == 200
    cif = (OBELIX_DIR / "cifs" / "jqc.cif").read_bytes()
    r = c.post("/simple", data={"action": "check", "cif": (io.BytesIO(cif), "x.cif")}, content_type="multipart/form-data")
    assert r.status_code == 200 and b"S/cm predicted" in r.data
    assert c.post("/simple", data={"action": "check"}, content_type="multipart/form-data").status_code == 400
    r = c.post("/simple", data={"action": "formula", "formula": "Li6PS5Cl F-43m"})
    assert r.status_code == 200 and b"CS Score" in r.data
    assert c.post("/simple", data={"action": "formula", "formula": "NaCl"}).status_code == 400
