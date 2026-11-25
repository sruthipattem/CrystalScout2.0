"""CrystalScout web platform (Flask).

Run:  python -m webapp.app   (then open http://127.0.0.1:5000)
"""
from __future__ import annotations

import io
import json
import os
import sys
import threading
import time
import uuid
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from flask import Flask, abort, g, jsonify, redirect, render_template, request, send_file, url_for

import warnings  # noqa: E402

try:  # saved models from another scikit-learn patch version are retrained automatically; hide the noise
    from sklearn.exceptions import InconsistentVersionWarning

    warnings.filterwarnings("ignore", category=InconsistentVersionWarning)
except ImportError:  # pragma: no cover
    pass

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from crystalscout.config import SUPERIONIC_THRESHOLD  # noqa: E402
from crystalscout.core.cif import CIFError  # noqa: E402
from crystalscout.core.descriptors import (FEATURE_INFO, FEATURES, SENDEK_2019_COEFS, SENDEK_2019_INTERCEPT,  # noqa: E402
                                           DescriptorError)
from crystalscout.core.structure import Structure  # noqa: E402
from crystalscout.data.materials_project import SNAPSHOT_INFO, load_pool_structures, load_snapshot  # noqa: E402
from crystalscout.data.obelix import build_features, load_obelix, training_table  # noqa: E402
from crystalscout.data.stage_a import StageACriteria  # noqa: E402
from crystalscout.featurize import featurize_cif_text, load_pool_features  # noqa: E402
from crystalscout.ml.benchmark import run_benchmark  # noqa: E402
from crystalscout.ml.explain import applicability, explain_sentence, logreg_contributions  # noqa: E402
from crystalscout.ml.design import (ANION_DOPANTS, DOPANTS, advise, assign_charges, lab_evaluate,  # noqa: E402
                                    synthesis_plan, to_formula)
from crystalscout.ml.insight import (FAMILY_LABELS, FormulaError, add_industry_columns, fmt_sigma,  # noqa: E402
                                     industry_flags, parse_checked_formula, sigma_readout, tier, verdict)
from crystalscout.ml.models import MODEL_LABELS, get_bundle  # noqa: E402
from crystalscout.ml.regression import (REG_LABELS, get_regressor, parse_formula, structure_volume_per_atom,  # noqa: E402
                                        volume_per_atom_from_dict)
from crystalscout.ml.structgen import assess, generate, structure_summary  # noqa: E402
from crystalscout.report import shortlist_pdf  # noqa: E402
from crystalscout.screening import (DEFAULT_RANK, RANK_LABELS, RANK_SHORT, ScreeningRun, list_runs,  # noqa: E402
                                    load_run, obelix_lookup, rank_column, run_screening, save_run, score_frame)

from . import charts  # noqa: E402
from .auth import ACCOUNTS, ROLES, init_auth  # noqa: E402

MAX_UPLOAD = 8 * 1024 * 1024
MODEL_KEYS = ("logreg", "random_forest", "gboost", "sendek2019")

PRESETS = [
    {"key": "standard", "name": "Standard screen", "icon": "target", "tone": "violet",
     "text": "Sendek-style defaults: stable, wide gap, no transition metals.",
     "gap": 1.0, "hull": 0.0, "tm": True, "excl": ""},
    {"key": "cost", "name": "Cost-aware", "icon": "factory", "tone": "amber",
     "text": "Drops Ge, Ga, In, Sc, Ta and rare earths for scalable supply.",
     "gap": 1.0, "hull": 0.0, "tm": True, "excl": "Ge Ga In Sc Ta Hf Y La Ce Nd Sm Gd Cs Rb Te"},
    {"key": "green", "name": "Low toxicity", "icon": "leaf", "tone": "good",
     "text": "Removes Pb, Cd, Hg, Tl, As, Be, Se and Sb.",
     "gap": 1.0, "hull": 0.0, "tm": True, "excl": "Pb Cd Hg Tl As Be Se Sb Ba"},
    {"key": "wide", "name": "High-voltage safe", "icon": "bolt", "tone": "blue",
     "text": "Band gap ≥ 3 eV for a wider electrochemical window.",
     "gap": 3.0, "hull": 0.0, "tm": True, "excl": ""},
    {"key": "broad", "name": "Broad exploration", "icon": "globe", "tone": "teal",
     "text": "Includes slightly metastable phases (E_hull ≤ 0.05 eV/atom).",
     "gap": 0.5, "hull": 0.05, "tm": True, "excl": ""},
]


class AppState:
    def __init__(self):
        self.lock = threading.Lock()
        self.bundle = None
        self._bench = None
        self._train = None
        self._default: Optional[ScreeningRun] = None
        self._consensus = None
        self.jobs: Dict[str, Dict] = {}
        self.uploads: Dict[str, Dict] = {}

    def ensure(self):
        with self.lock:
            if self.bundle is None:
                self.bundle = get_bundle()
        return self.bundle

    @property
    def reg(self):
        return get_regressor(verbose=False)

    @property
    def bench(self):
        if self._bench is None:
            self._bench = run_benchmark(self.ensure())
        return self._bench

    @property
    def train(self) -> pd.DataFrame:
        if self._train is None:
            self._train = training_table()
        return self._train

    @property
    def consensus(self):
        if self._consensus is None:
            from crystalscout.ml.consensus import evaluate

            self._consensus = evaluate(self.ensure(), self.reg)
        return self._consensus

    def default_run(self) -> ScreeningRun:
        if self._default is not None:
            return self._default
        for r in list_runs():
            if r["source"] == "snapshot" and r["criteria_text"] == StageACriteria().describe() \
                    and r.get("rank_model") == DEFAULT_RANK:
                run = load_run(r["run_id"])
                if run is not None:
                    self._default = run
                    return run
        self._default = run_screening(self.ensure())
        return self._default

    @property
    def ref(self) -> pd.DataFrame:
        return self.default_run().shortlist


STATE = AppState()


def fnum(v, d=2):
    from jinja2 import Undefined

    if isinstance(v, Undefined):
        return "—"
    try:
        if v is None or (isinstance(v, float) and np.isnan(v)):
            return "—"
        return f"{float(v):.{d}f}"
    except (TypeError, ValueError):
        return str(v)


def fsci(v):
    try:
        v = float(v)
        if np.isnan(v):
            return "—"
        m, e = f"{v:.1e}".split("e")
        return f"{m} × 10<sup>{int(e)}</sup>"
    except (TypeError, ValueError):
        return "—"


def flog(v):
    """log10 σ → '3.2 × 10⁻⁴' HTML."""
    try:
        v = float(v)
        if np.isnan(v):
            return "—"
        return fsci(10 ** v)
    except (TypeError, ValueError):
        return "—"


def formula_html(f: str) -> str:
    import re
    return re.sub(r"(\d+(?:\.\d+)?)", r"<sub>\1</sub>", str(f))


def isnum(v) -> bool:
    try:
        return v is not None and not np.isnan(float(v))
    except (TypeError, ValueError):
        return False


def _secret_key() -> str:
    env = os.environ.get("CRYSTALSCOUT_SECRET")
    if env:
        return env
    from crystalscout.config import DATA_DIR
    import secrets

    f = DATA_DIR / ".secret_key"
    if not f.exists():
        f.write_text(secrets.token_hex(32))
    return f.read_text().strip()


def create_app() -> Flask:
    app = Flask(__name__)
    app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD
    app.secret_key = _secret_key()
    init_auth(app)
    app.jinja_env.filters.update(f=fnum, sci=fsci, slog=flog, formula=formula_html)
    app.jinja_env.tests["num"] = isnum
    app.jinja_env.globals.update(FEATURES=FEATURES, FEATURE_INFO=FEATURE_INFO, MODEL_LABELS=MODEL_LABELS,
                                 SNAPSHOT_INFO=SNAPSHOT_INFO, RANK_LABELS=RANK_LABELS, RANK_SHORT=RANK_SHORT,
                                 FAMILY_LABELS=FAMILY_LABELS, tier=tier)

    # ---------------------------------------------------------------- helpers
    def get_run(run_id: Optional[str]) -> ScreeningRun:
        if run_id:
            run = load_run(run_id)
            if run is None:
                abort(404)
            return run
        return STATE.default_run()

    def has_desc(row: Dict) -> bool:
        return all(isnum(row.get(k)) for k in FEATURES)

    def material_view(row: Dict, structure: Optional[Structure], run_id: Optional[str], comp: Optional[Dict] = None,
                      vpa: float = np.nan, space_group=None) -> Dict:
        b = STATE.ensure()
        formula = str(row.get("formula", ""))
        comp = comp or parse_formula(formula)
        ctx = {"row": row, "run_id": run_id, "has_desc": has_desc(row), "obelix": obelix_lookup(formula),
               "viewer": json.dumps(structure.viewer_payload()) if structure is not None else None,
               "flags": industry_flags(comp), "probs": None}
        p_lr = None
        if ctx["has_desc"]:
            x = np.array([float(row[k]) for k in FEATURES])
            expl = logreg_contributions(b, x)
            probs = {m: float(b.models[m].predict_proba(x[None, :])[0, 1]) for m in MODEL_KEYS}
            p_lr = probs["logreg"]
            tr = STATE.train
            strips = {}
            for k in FEATURES:
                lo, hi = float(tr[k].quantile(0.01)), float(tr[k].quantile(0.99))
                strips[k] = charts.strip(k, tr.loc[tr.label == 1, k].tolist(), tr.loc[tr.label == 0, k].tolist(),
                                         float(row[k]), lo, hi)
            ctx.update(probs=probs, expl=expl, why=explain_sentence(expl), domain=applicability(b, x),
                       contrib_svg=charts.contributions(expl["terms"]), strips=strips)
        sg = space_group or row.get("space_group") or (structure.meta.get("space_group") if structure is not None else None)
        ctx["sigma"] = sigma_readout(STATE.reg, comp, sg, vpa, row if ctx["has_desc"] else None, formula, p_lr, STATE.ref)
        ctx["verdict"] = verdict(p_lr, ctx["sigma"]["log10_sigma"], ctx["sigma"].get("sigma_pct"))
        return ctx

    def view_rows(run: ScreeningRun) -> pd.DataFrame:
        sl = add_industry_columns(run.shortlist)
        return sl

    # ------------------------------------------------------------------ dashboard
    @app.route("/")
    def overview():
        b = STATE.ensure()
        run = STATE.default_run()
        s = run.summary()
        sl = view_rows(run)
        sig_pct = sl["log10_sigma_pred"].rank(pct=True)
        strong = sl[(sl["p_logreg"] >= 0.5) & ((sl["log10_sigma_pred"] >= -4) | (sig_pct >= 0.75))]
        top = sl.head(8).to_dict("records")
        fam = sl.head(50)["family"].value_counts()
        fam_bars = charts.hbars([(FAMILY_LABELS[k], int(v)) for k, v in fam.items()], fmt="{:.0f}", label_w=150)
        rep = b.report
        reg_rep = STATE.reg.report
        pos_ids = {r["material_id"] for r in STATE.bench["table"] if r["superionic_dftmd"] == 1}
        hits = sl[sl["material_id"].isin(pos_ids)][["formula", "rank", "material_id"]].to_dict("records")
        kpi = {"scanned": s["n_input"], "passed": s["n_stage_a"], "strong": int(len(strong)),
               "new": int(sl.head(50)["obelix_sigma"].isna().sum())}
        funnel = charts.funnel(s["funnel"] + [("ranked by CrystalScout Score", s["n_scored"])])
        return render_template("overview.html", run=run, summary=s, top=top, kpi=kpi, fam_bars=fam_bars, rep=rep,
                               reg=reg_rep, bench=STATE.bench, hits=hits, funnel=funnel)

    # ------------------------------------------------------------------ screen
    def screen_page(c, error=None, status=200, preset="standard"):
        return render_template("screen.html", c=c, runs=list_runs()[:10], has_key=bool(os.environ.get("MP_API_KEY")),
                               error=error, presets=PRESETS, preset=preset), status

    @app.route("/screen", methods=["GET", "POST"])
    def screen():
        if request.method == "GET":
            return screen_page(StageACriteria())
        f = request.form
        try:
            excl = tuple(e.strip().capitalize() for e in f.get("exclude_elements", "").replace(",", " ").split() if e.strip())
            c = StageACriteria(band_gap_min=float(f.get("band_gap_min", 1.0)), ehull_max=float(f.get("ehull_max", 0.0)),
                               exclude_transition_metals=f.get("exclude_tm") == "on", exclude_elements=excl,
                               max_nsites=int(f.get("max_nsites") or 0))
        except ValueError:
            return screen_page(StageACriteria(), "Thresholds must be numbers.", 400)
        source = f.get("source", "snapshot")
        rank = f.get("rank_model", DEFAULT_RANK)
        if rank not in RANK_LABELS:
            rank = DEFAULT_RANK
        if source == "live" and not g.user["can_live"]:
            return screen_page(c, "Expo guest accounts can't query the live Materials Project. Use the bundled snapshot.", 403)
        key = f.get("api_key", "").strip() or None
        if source == "snapshot" and (c.band_gap_min < 0.5 or c.ehull_max > 0.1):
            return screen_page(c, "The bundled snapshot holds structures for band gap ≥ 0.5 eV and E_hull ≤ 0.1 eV/atom. "
                                  "Tighten the thresholds or switch to the live Materials Project.", 400)
        job_id = uuid.uuid4().hex[:10]
        STATE.jobs[job_id] = {"status": "running", "stage": "starting", "done": 0, "total": 0, "run_id": None, "error": None,
                              "source": source, "user": g.user["username"]}
        run_by = g.user["username"]

        def work():
            job = STATE.jobs[job_id]

            def prog(stage, a, b):
                job.update(stage=stage, done=a, total=b)
            try:
                job.update(stage="filter")
                run = run_screening(STATE.ensure(), c, source=source, api_key=key, progress=prog, rank_model=rank, save=False)
                run.stats["run_by"] = run_by
                save_run(run)
                job.update(status="done", run_id=run.run_id)
            except Exception as exc:  # noqa: BLE001
                job.update(status="error", error=str(exc))

        threading.Thread(target=work, daemon=True).start()
        return redirect(url_for("job", job_id=job_id))

    def own_job(job_id):
        j = STATE.jobs.get(job_id)
        if not j or (j.get("user") != g.user["username"] and not g.user["is_admin"]):
            abort(404)
        return j

    @app.route("/jobs/<job_id>")
    def job(job_id):
        own_job(job_id)
        return render_template("job.html", job_id=job_id, job=STATE.jobs[job_id])

    @app.route("/api/jobs/<job_id>")
    def job_api(job_id):
        j = own_job(job_id)
        out = dict(j)
        if j.get("run_id"):
            out["url"] = url_for("shortlist", run_id=j["run_id"])
        return jsonify(out)

    # ------------------------------------------------------------------ shortlist
    @app.route("/shortlist")
    @app.route("/runs/<run_id>")
    def shortlist(run_id=None):
        run = get_run(run_id)
        a = request.args
        model = a.get("model", run.rank_model)
        if model not in RANK_LABELS:
            model = run.rank_model if run.rank_model in RANK_LABELS else DEFAULT_RANK
        domain, q, fam = a.get("domain", "all"), a.get("q", "").strip(), a.get("family", "")
        no_risk, no_tox, struct_only = a.get("no_risk") == "1", a.get("no_tox") == "1", a.get("struct") == "1"
        sl = view_rows(run)
        col = rank_column(model)
        sl = sl.sort_values([col, "log10_sigma_pred"], ascending=False, na_position="last").reset_index(drop=True)
        sl["view_rank"] = np.arange(1, len(sl) + 1)
        fam_counts = sl["family"].value_counts().to_dict()
        if domain == "in":
            sl = sl[sl["in_domain"].astype(str) == "True"]
        if fam:
            sl = sl[sl["family"] == fam]
        if no_risk:
            sl = sl[sl["supply_risk"] == ""]
        if no_tox:
            sl = sl[sl["toxic"] == ""]
        if struct_only:
            sl = sl[sl["p_logreg"].notna()]
        if q:
            sl = sl[sl["formula"].str.contains(q, case=False, regex=False) | sl["material_id"].str.contains(q, case=False, regex=False)]
        try:
            limit = max(10, min(int(a.get("limit", 100)), 2000))
        except ValueError:
            limit = 100
        return render_template("shortlist.html", run=run, summary=run.summary(), rows=sl.head(limit).to_dict("records"),
                               n_rows=len(sl), model=model, domain=domain, q=q, limit=limit, fam=fam, fam_counts=fam_counts,
                               no_risk=no_risk, no_tox=no_tox, struct_only=struct_only, runs=list_runs()[:20])

    @app.route("/compare")
    def compare():
        run = get_run(request.args.get("run"))
        ids = [i for i in request.args.get("ids", "").split(",") if i][:4]
        sl = view_rows(run)
        sl["sigma_pct"] = sl["log10_sigma_pred"].rank(pct=True)
        sl = sl.set_index("material_id")
        items = []
        for mid in ids:
            if mid not in sl.index:
                continue
            r = sl.loc[mid].to_dict()
            r["material_id"] = mid
            r["tier"] = tier(r.get("log10_sigma_pred"))
            r["verdict"] = verdict(r.get("p_logreg") if isnum(r.get("p_logreg")) else None, r.get("log10_sigma_pred"),
                                   r.get("sigma_pct"))
            items.append(r)
        if len(items) < 2:
            return render_template("message.html", title="Pick at least two materials",
                                   text="Tick 2 to 4 materials in the shortlist, then press Compare."), 400
        chart = charts.sigma_intervals([{"label": it["formula"], "mu": it["log10_sigma_pred"], "lo": it["log10_sigma_lo"],
                                         "hi": it["log10_sigma_hi"]} for it in items])
        return render_template("compare.html", items=items, run=run, chart=chart)

    @app.route("/runs/<run_id>/rejected")
    def rejected(run_id):
        run = get_run(run_id)
        rej = run.rejected
        reason = request.args.get("reason", "")
        q = request.args.get("q", "").strip()
        view = rej
        if reason:
            view = view[view["rejection_reasons"].str.contains(reason, regex=False)]
        if q:
            view = view[view["formula"].astype(str).str.contains(q, case=False, regex=False)]
        return render_template("rejected.html", run=run, rows=view.head(300).to_dict("records"), n=len(view),
                               counts=run.stats.get("fail_counts", {}), reason=reason, q=q,
                               bars=charts.hbars(sorted(run.stats.get("fail_counts", {}).items(), key=lambda kv: -kv[1]),
                                                 fmt="{:,.0f}", var="var(--s2)", label_w=190))

    @app.route("/runs/<run_id>/candidates.csv")
    def run_csv(run_id):
        run = get_run(run_id)
        buf = io.BytesIO(add_industry_columns(run.shortlist).to_csv(index=False).encode())
        return send_file(buf, mimetype="text/csv", as_attachment=True, download_name=f"crystalscout_{run_id}_ranked.csv")

    @app.route("/runs/<run_id>/rejections.csv")
    def run_rej_csv(run_id):
        run = get_run(run_id)
        buf = io.BytesIO(run.rejected.to_csv(index=False).encode())
        return send_file(buf, mimetype="text/csv", as_attachment=True, download_name=f"crystalscout_{run_id}_rejections.csv")

    @app.route("/runs/<run_id>/report.pdf")
    def run_pdf(run_id):
        run = get_run(run_id)
        try:
            n = max(5, min(int(request.args.get("n", 25)), 100))
        except ValueError:
            n = 25
        pdf = shortlist_pdf(run, STATE.ensure().report, top_n=n)
        return send_file(io.BytesIO(pdf), mimetype="application/pdf", as_attachment=True,
                         download_name=f"crystalscout_{run_id}_shortlist.pdf")

    # ------------------------------------------------------------------ material page
    @app.route("/material/<mp_id>")
    def material(mp_id):
        run_id = request.args.get("run")
        run = get_run(run_id) if run_id else STATE.default_run()
        row = None
        m = run.shortlist[run.shortlist["material_id"] == mp_id]
        if len(m):
            row = m.iloc[0].to_dict()
        if row is None:
            pool = load_pool_features()
            snap = load_snapshot()
            s = snap[snap["material_id"] == mp_id]
            if not len(s):
                abort(404)
            p = pool[pool["material_id"] == mp_id]
            row = {**s.iloc[0].to_dict(), **(p.iloc[0].to_dict() if len(p) else {})}
        sd = load_pool_structures().get(mp_id)
        st = Structure.from_pymatgen_dict(sd, name=mp_id) if sd else None
        vpa = volume_per_atom_from_dict(sd) if sd else np.nan
        ctx = material_view(row, st, run.run_id if run_id else None, vpa=vpa)
        ctx.update(title=row.get("formula"), subtitle=mp_id, kind="mp",
                   cif_url=url_for("material_cif", mp_id=mp_id) if st else None,
                   mp_url=f"https://next-gen.materialsproject.org/materials/{mp_id}",
                   rank=row.get("rank"), n_ranked=len(run.shortlist))
        return render_template("material.html", **ctx)

    @app.route("/material/<mp_id>.cif")
    def material_cif(mp_id):
        sd = load_pool_structures().get(mp_id)
        if not sd:
            abort(404)
        st = Structure.from_pymatgen_dict(sd, name=mp_id)
        return send_file(io.BytesIO(st.to_cif().encode()), mimetype="chemical/x-cif", as_attachment=True,
                         download_name=f"{mp_id}.cif")

    # ------------------------------------------------------------------ quick check
    def cif_link(formula: str, sg: Optional[str] = None, base: Optional[str] = None) -> str:
        args = {"formula": formula}
        if sg:
            args["sg"] = sg
        if base and base != formula:
            args["base"] = base
        return url_for("design_cif", **args)

    def formula_estimate(text: str, base: Optional[str] = None) -> Dict:
        """Formula in, full answer out: existence, feasibility, a crystal structure and the structure-path prediction."""
        parts = text.strip().split()
        formula = parts[0] if parts else ""
        sg = " ".join(parts[1:]) or None
        a = assess(STATE.reg, formula, sg, base, STATE.ref, STATE.ensure())
        comp, rd = a["comp"], a["sigma"]
        return {"formula": formula, "space_group": sg or a["space_group_used"], "sigma": rd, "flags": industry_flags(comp),
                "verdict": verdict(a["p_logreg"], rd["log10_sigma"], rd.get("sigma_pct")), "obelix": obelix_lookup(formula),
                "existence": a["existence"], "feasibility": a["feasibility"], "confidence": a["confidence"],
                "structure": structure_summary(a["structure"]), "p_logreg": a["p_logreg"],
                "sigma_comp": a["sigma_composition_only"], "honesty": a["honesty"],
                "cif_url": cif_link(formula, sg, base) if a["structure"] and a["structure"].get("ok") else None}

    @app.route("/design/cif")
    def design_cif():
        """A crystal structure (CIF) for any formula: the real one when known, else a substitution model."""
        f = request.args.get("formula", "").strip()
        try:
            parse_checked_formula(f)
        except FormulaError as exc:
            abort(400, description=str(exc))
        gen = generate(f, request.args.get("base", "").strip() or None, space_group=request.args.get("sg", "").strip() or None)
        if not gen.get("ok"):
            abort(400, description="No crystal structure could be built for " + f + ": " + gen.get("reason", ""))
        safe = "".join(ch for ch in f if ch.isalnum() or ch in "._-")
        return send_file(io.BytesIO(gen["cif"].encode()), mimetype="chemical/x-cif", as_attachment=True,
                         download_name=f"{safe}_CrystalScout.cif")

    def score_cif(name: str, text: str) -> Dict:
        st, feats = featurize_cif_text(text, name=name)
        row = {"formula": st.formula(), "material_id": name, **feats}
        ctx = material_view(row, st, None, comp=st.composition(), vpa=structure_volume_per_atom(st))
        return {"name": name, "ok": True, "row": row, "probs": ctx["probs"], "why": ctx.get("why", ""),
                "domain": ctx.get("domain", {}), "sigma": ctx["sigma"], "verdict": ctx["verdict"], "flags": ctx["flags"],
                "obelix": ctx["obelix"], "cif": text}

    @app.route("/simple", methods=["GET", "POST"])
    def simple():
        """Expo-friendly page: one formula or one CIF in, a clear answer out."""
        result, error, est = None, None, None
        formula = ""
        if request.method == "POST":
            action = request.form.get("action")
            if action == "formula":
                formula = request.form.get("formula", "").strip()
                try:
                    est = formula_estimate(formula)
                except FormulaError as exc:
                    error = str(exc)
            elif action == "check":
                fs = request.files.get("cif")
                text = fs.read().decode("utf-8", errors="replace") if fs and fs.filename else ""
                if not text.strip():
                    error = "Choose a .cif file first."
                else:
                    try:
                        it = score_cif(fs.filename, text)
                        token = uuid.uuid4().hex[:10]
                        STATE.uploads[token] = {"kind": "cif", "user": g.user["username"], "items": [it]}
                        result = {**it, "url": url_for("predicted", token=token, i=0)}
                    except (CIFError, DescriptorError, ValueError) as exc:
                        error = f"That file couldn't be read as a crystal structure: {exc}"
        run = STATE.default_run()
        top = view_rows(run).head(6).to_dict("records")
        examples = ["Li6PS5Cl", "Li10GeP2S12", "Li7La3Zr2O12 Ia-3d", "Li3InCl6 C2/m", "Li3PO4", "Li2B2S5"]
        return render_template("simple.html", top=top, run=run, result=result, error=error, est=est, formula=formula,
                               examples=examples), (400 if error else 200)

    # ------------------------------------------------------------------ score materials
    @app.route("/predict", methods=["GET", "POST"])
    def predict():
        if request.method == "GET":
            text = request.args.get("formula", "").strip()
            if text:
                rows, errs = [], []
                for line in text.replace(";", "\n").replace(",", "\n").splitlines():
                    if line.strip():
                        try:
                            rows.append(formula_estimate(line))
                        except FormulaError as exc:
                            errs.append(f"{line.strip()}: {exc}")
                return render_template("predict.html", tab="formula", formula_rows=rows, formula_errors=errs,
                                       formula_text=text, error=None)
            return render_template("predict.html", tab=request.args.get("tab", "cif"), error=None)
        kind = request.form.get("kind", "cif")
        if kind == "formula":
            text = request.form.get("formulas", "")
            rows, errs = [], []
            for line in text.splitlines()[:100]:
                if line.strip():
                    try:
                        rows.append(formula_estimate(line))
                    except FormulaError as exc:
                        errs.append(f"{line.strip()}: {exc}")
            if not rows and not errs:
                return render_template("predict.html", tab="formula", error="Enter at least one formula."), 400
            rows.sort(key=lambda r: -r["sigma"]["log10_sigma"])
            return render_template("predict.html", tab="formula", formula_rows=rows, formula_errors=errs,
                                   formula_text=text, error=None)
        if kind == "csv":
            csv_file = request.files.get("csv")
            if not csv_file or not csv_file.filename:
                return render_template("predict.html", tab="csv", error="Choose a CSV file."), 400
            try:
                df = pd.read_csv(csv_file)
            except Exception as exc:  # noqa: BLE001
                return render_template("predict.html", tab="csv", error=f"Could not read the CSV: {exc}"), 400
            missing = [k for k in FEATURES if k not in df.columns]
            if missing:
                return render_template("predict.html", tab="csv",
                                       error="The CSV needs the columns " + ", ".join(FEATURES) + f". Missing: {', '.join(missing)}."), 400
            if "formula" not in df.columns:
                df["formula"] = [f"row {i + 1}" for i in range(len(df))]
            df["material_id"] = df["material_id"] if "material_id" in df.columns else [f"row-{i + 1}" for i in range(len(df))]
            scored = score_frame(STATE.ensure(), df).sort_values("p_logreg", ascending=False)
            token = uuid.uuid4().hex[:10]
            STATE.uploads[token] = {"kind": "csv", "rows": scored.to_dict("records"), "user": g.user["username"]}
            return render_template("predict.html", tab="csv", csv_rows=scored.to_dict("records"), token=token, error=None)
        items = []
        for fs in request.files.getlist("cifs"):
            if fs and fs.filename:
                items.append((fs.filename, fs.read().decode("utf-8", errors="replace")))
        pasted = request.form.get("cif_text", "").strip()
        if pasted:
            items.append(("pasted.cif", pasted))
        if not items:
            return render_template("predict.html", tab="cif", error="Choose at least one CIF file or paste CIF text."), 400
        token = uuid.uuid4().hex[:10]
        store = []
        for name, text in items[:50]:
            try:
                store.append(score_cif(name, text))
            except (CIFError, DescriptorError, ValueError) as exc:
                store.append({"name": name, "ok": False, "error": str(exc)})
        STATE.uploads[token] = {"kind": "cif", "items": store, "user": g.user["username"]}
        results = sorted(enumerate(store), key=lambda t: -(t[1]["sigma"].get("cs_score", 0) if t[1]["ok"] else -1))
        return render_template("predict.html", tab="cif", results=results, token=token, error=None)

    @app.route("/predict/<token>/<int:i>")
    def predicted(token, i):
        up = STATE.uploads.get(token)
        if up and up.get("user") != g.user["username"]:
            up = None
        if not up or up["kind"] != "cif" or i >= len(up["items"]) or not up["items"][i]["ok"]:
            abort(404)
        it = up["items"][i]
        st, _ = featurize_cif_text(it["cif"], name=it["name"])
        ctx = material_view(it["row"], st, None, comp=st.composition(), vpa=structure_volume_per_atom(st))
        ctx.update(title=it["row"]["formula"], subtitle=it["name"], kind="upload", cif_url=None, mp_url=None,
                   rank=None, n_ranked=None)
        return render_template("material.html", **ctx)

    @app.route("/predict/<token>.csv")
    def predicted_csv(token):
        up = STATE.uploads.get(token)
        if not up or up.get("user") != g.user["username"]:
            abort(404)
        if up["kind"] == "csv":
            df = pd.DataFrame(up["rows"])
        else:
            recs = []
            for it in up["items"]:
                if not it["ok"]:
                    recs.append({"file": it["name"], "error": it["error"]})
                    continue
                s = it["sigma"]
                recs.append({"file": it["name"], "formula": it["row"]["formula"], "crystalscout_score": s.get("cs_score"),
                             "sigma_pred_S_cm": s["sigma_S_cm"], "sigma_lo_S_cm": 10 ** s["log10_lo"],
                             "sigma_hi_S_cm": 10 ** s["log10_hi"], "tier": s["tier"]["name"],
                             **{k: it["row"][k] for k in FEATURES}, **{f"p_{m}": v for m, v in (it["probs"] or {}).items()},
                             "in_domain": it["domain"].get("in_domain"), "verdict": it["verdict"]["label"], "why": it["why"]})
            df = pd.DataFrame(recs)
        return send_file(io.BytesIO(df.to_csv(index=False).encode()), mimetype="text/csv", as_attachment=True,
                         download_name="crystalscout_predictions.csv")

    # ------------------------------------------------------------------ JSON (in-app)
    @app.route("/api/predict", methods=["POST"])
    def api_predict():
        data = request.get_json(silent=True) or {}
        try:
            x = np.array([float(data[k]) for k in FEATURES])
        except (KeyError, TypeError, ValueError):
            return jsonify({"error": "send JSON with numeric " + ", ".join(FEATURES)}), 400
        b = STATE.ensure()
        expl = logreg_contributions(b, x)
        return jsonify({"probabilities": {m: float(b.models[m].predict_proba(x[None, :])[0, 1]) for m in MODEL_LABELS},
                        "explanation": expl, "applicability": applicability(b, x)})

    @app.route("/api/estimate")
    def api_estimate():
        try:
            e = formula_estimate(request.args.get("formula", ""))
        except FormulaError as exc:
            return jsonify({"error": str(exc)}), 400
        s = e["sigma"]
        return jsonify({"formula": e["formula"], "log10_sigma": s["log10_sigma"], "sigma_S_cm": s["sigma_S_cm"],
                        "interval_80": [10 ** s["log10_lo"], 10 ** s["log10_hi"]], "tier": s["tier"]["name"],
                        "crystalscout_score": s.get("cs_score"), "flags": e["flags"]})

    # ------------------------------------------------------------------ descriptor lab
    @app.route("/explainer")
    def explainer():
        b = STATE.ensure()
        lr = b.models["logreg"]
        sc, clf = lr.named_steps["scale"], lr.named_steps["clf"]
        tr = STATE.train
        ranges = {k: [float(tr[k].quantile(0.0)), float(tr[k].quantile(1.0))] for k in FEATURES}
        presets = []
        for f in ["Li10GeP2S12", "Li6PS5Cl", "Li7La3Zr2O12", "Li3InCl6", "Li3PO4", "LiTi2(PO4)3"]:
            m = tr[tr["formula"] == f]
            if len(m):
                r = m.iloc[0]
                presets.append({"name": f, "sigma": float(r["sigma"]), **{k: round(float(r[k]), 3) for k in FEATURES}})
        pool = load_pool_features().set_index("material_id")
        for mid, nm in [("mp-532413", "Li5B7S13 (MP)"), ("mp-29410", "Li2B2S5 (MP)")]:
            if mid in pool.index:
                r = pool.loc[mid]
                presets.append({"name": nm, "sigma": None, **{k: round(float(r[k]), 3) for k in FEATURES}})
        model = {"mean": sc.mean_.tolist(), "scale": sc.scale_.tolist(), "coef": clf.coef_[0].tolist(),
                 "intercept": float(clf.intercept_[0]), "sendek": [SENDEK_2019_COEFS[k] for k in FEATURES],
                 "sendek_intercept": SENDEK_2019_INTERCEPT, "features": FEATURES, "ranges": ranges,
                 "info": FEATURE_INFO, "presets": presets}
        return render_template("explainer.html", model=json.dumps(model), presets=presets)

    # ------------------------------------------------------------------ model validation
    @app.route("/models")
    def models():
        b = STATE.ensure()
        rep = b.report
        keys = list(MODEL_KEYS)
        labels = {k: rep["models"][k]["label"] for k in keys}
        roc_test = charts.roc_chart({k: rep["models"][k]["test"]["roc_curve"] for k in keys}, labels,
                                    {k: rep["models"][k]["test"]["roc_auc"] for k in keys}, "ROC on the OBELiX test split")
        roc_cv = charts.roc_chart({k: rep["models"][k]["cv_all_roc_curve"] for k in keys}, labels,
                                  {k: rep["models"][k]["cv_all_metrics"]["roc_auc"] for k in keys},
                                  "ROC, leave-family-out cross-validation")
        eq = rep["logreg_equation"]
        coef = charts.coef_compare(eq["coef_raw"], {k: eq["sendek2019"][k] for k in FEATURES})
        imp = rep["permutation_importance_test_auc"]["logreg"]
        reg = STATE.reg.report
        parity = charts.parity(reg["test_predictions"])
        feat_bars = charts.hbars([(t["feature"], t["importance"] * 100) for t in reg["top_features"][:10]],
                                 fmt="{:.1f} %", label_w=170, var="var(--s1)")
        from crystalscout.ml.validation import load_validation

        return render_template("models.html", rep=rep, keys=keys, roc_test=roc_test, roc_cv=roc_cv, coef=coef, eq=eq,
                               bench=STATE.bench, imp=imp, reg=reg, REG_LABELS=REG_LABELS, parity=parity,
                               feat_bars=feat_bars, cons=STATE.consensus, hold=load_validation())

    # ------------------------------------------------------------------ data & methods
    @app.route("/data")
    def data():
        df = load_obelix()
        feats = build_features()
        snap = load_snapshot()
        hist = charts.histogram(df["log10_sigma"].dropna().tolist(), 28, -14, 0, threshold=np.log10(SUPERIONIC_THRESHOLD),
                                xlabel="log₁₀ room-temperature ionic conductivity (S/cm)")
        fam = feats[feats.excluded_reason == ""].groupby("family")["label"].agg(["count", "sum"]).sort_values("count", ascending=False)
        flags = {
            "Conductivity is an upper bound (\"<1E-10\")": int(df["sigma_upper_bound"].sum()),
            "CIF from a closely related compound": int(feats["close_match"].sum()),
            "Disordered / partial-occupancy CIF": int(feats["disordered"].fillna(False).astype(bool).sum()),
            "CIF Li content differs > 10 % from reported": int(feats["li_mismatch"].fillna(False).astype(bool).sum()),
            "CIF contains no Li (excluded)": int(feats["no_li_in_cif"].fillna(False).astype(bool).sum()),
        }
        q = request.args.get("q", "").strip()
        view = feats
        if q:
            view = view[view["formula"].str.contains(q, case=False, regex=False) | view["family"].str.contains(q, case=False, regex=False)]
        return render_template("data.html", n_all=len(df), n_cif=int(df.has_cif.sum()), n_pos=int(df.label.sum()),
                               n_used=int((feats.excluded_reason == "").sum()), hist=hist, fam=fam.reset_index().to_dict("records"),
                               flags=flags, rows=view.sort_values("sigma", ascending=False).head(400).to_dict("records"),
                               q=q, n_snap=len(snap), n_pool=len(load_pool_features()), n_reg=STATE.reg.report["protocol"]["n_total"])

    @app.route("/methods")
    def methods():
        return render_template("methods.html", rep=STATE.ensure().report, reg=STATE.reg.report)

    @app.route("/developers")
    def developers():
        try:
            import fastapi  # noqa: F401
            has_fastapi = True
        except ImportError:
            has_fastapi = False
        return render_template("developers.html", has_fastapi=has_fastapi, host=request.host.split(":")[0])

    # ------------------------------------------------------------------ DESIGN: Formula Lab
    def swap_options(comp: Dict) -> Dict[str, List[str]]:
        ch = assign_charges(comp) or {}
        out = {}
        for e in comp:
            q = ch.get(e)
            if e == "Li" or q is None:
                continue
            pool = DOPANTS if q > 0 else ANION_DOPANTS
            opts = []
            for qq in (q, q - 1, q + 1):
                opts += [d for d in pool.get(qq, ()) if d != e and d not in opts]
            out[e] = opts
        return out

    @app.route("/lab")
    def lab():
        f = request.args.get("formula", "Li6PS5Cl").strip() or "Li6PS5Cl"
        sg = request.args.get("sg", "F-43m" if f == "Li6PS5Cl" else "").strip()
        return render_template("lab.html", formula=f, sg=sg, base=request.args.get("base", f))

    @app.route("/api/lab")
    def api_lab():
        f = request.args.get("formula", "").strip()
        sg = request.args.get("sg", "").strip() or None
        base = request.args.get("base", "").strip() or None
        if base:
            try:
                parse_checked_formula(base)
            except FormulaError:
                base = None
        try:
            r = lab_evaluate(STATE.reg, f, sg, base, STATE.ref, STATE.ensure())
        except FormulaError as exc:
            return jsonify({"error": str(exc)}), 400
        comp = parse_formula(f)
        s = r["sigma"]
        st = structure_summary(r["structure"])
        sc = r["sigma_composition_only"]
        return jsonify({
            "existence": r["existence"], "feasibility": r["feasibility"], "confidence": r["confidence"],
            "structure": st, "cif_url": cif_link(f, sg, base) if st else None, "has_structure": s["has_structure"],
            "sigma_comp_text": sc["sigma_text"] if sc else None, "p_logreg": r["p_logreg"],
            "compare_path": r.get("compare_path"), "descriptor_change": r.get("descriptor_change"), "honesty": r["honesty"],
            "formula": r["formula"], "pretty": r["pretty"], "space_group": r["space_group"],
            "log10_sigma": s["log10_sigma"], "sigma_text": s["sigma_text"], "lo_text": s["lo_text"], "hi_text": s["hi_text"],
            "gauge": [s["gauge_lo"], s["gauge_pct"], s["gauge_hi"]], "tier": s["tier"], "p_trees": s["p_trees_superionic"],
            "cs_score": s.get("cs_score"), "beats": s.get("cs_beats"), "verdict": r["verdict"], "flags": r["flags"],
            "known": s["known_composition"], "obelix": obelix_lookup(f), "charge_balanced": r["charge_balanced"],
            "oxidation": r["oxidation"], "base": r.get("base"), "delta_log": r.get("delta_log"), "gain": r.get("gain"),
            "drivers": r.get("drivers", []), "elements": [[e, comp[e]] for e in comp], "swaps": swap_options(comp)})

    # ------------------------------------------------------------------ DESIGN: Dopant Advisor
    @app.route("/advisor", methods=["GET", "POST"])
    def advisor():
        src = request.form if request.method == "POST" else request.args
        f = src.get("formula", "").strip()
        sg = src.get("sg", "").strip()
        opts = {"allow_tm": src.get("allow_tm") == "on", "avoid_toxic": src.get("avoid_toxic", "on") == "on",
                "avoid_supply_risk": src.get("avoid_supply_risk") == "on"}
        if request.method == "GET" and not f:
            return render_template("advisor.html", res=None, formula="", sg="", opts={"allow_tm": False, "avoid_toxic": True,
                                                                                      "avoid_supply_risk": False}, error=None)
        try:
            res = advise(STATE.reg, f, sg or None, top=10, ref=STATE.ref, bundle=STATE.ensure(), **opts)
        except FormulaError as exc:
            return render_template("advisor.html", res=None, formula=f, sg=sg, opts=opts, error=str(exc)), 400
        if not res["suggestions"]:
            rj = res.get("rejected", {})
            msg = ("No new dopings passed the chemistry checks for this formula "
                   f"({res['n_variants']} charge-neutral variants generated; {rj.get('size', 0)} rejected for ion size, "
                   f"{rj.get('site', 0)} for site fit). Try allowing transition metals or another composition.")
            return render_template("advisor.html", res=None, formula=f, sg=sg, opts=opts, error=msg), 400
        recipe = synthesis_plan(res["best"]["formula"], 5.0)
        chart = charts.hbars([(v["formula"], 10 ** v["log10_sigma"] * 1000) for v in res["all"][:12]], fmt="{:.2f} mS/cm",
                             label_w=230, var="var(--s1)")
        sg_used = sg or res["space_group"]
        return render_template("advisor.html", res=res, formula=f, sg=sg, opts=opts, error=None, recipe=recipe, chart=chart,
                               cif=lambda x: cif_link(x, sg_used or None, f))

    @app.route("/api/advise")
    def api_advise():
        try:
            res = advise(STATE.reg, request.args.get("formula", ""), request.args.get("sg") or None, top=int(request.args.get("top", 10)),
                         ref=STATE.ref, bundle=STATE.ensure(), allow_tm=request.args.get("allow_tm") == "1",
                         avoid_toxic=request.args.get("avoid_toxic", "1") == "1",
                         avoid_supply_risk=request.args.get("avoid_supply_risk") == "1")
        except (FormulaError, ValueError) as exc:
            return jsonify({"error": str(exc)}), 400
        res.pop("all", None)
        res["base"] = {k: res["base"][k] for k in ("log10_sigma", "sigma_text", "lo_text", "hi_text", "tier")}
        return jsonify(res)

    # ------------------------------------------------------------------ DESIGN: Synthesis Planner
    @app.route("/recipe")
    def recipe():
        f = request.args.get("formula", "").strip()
        if not f:
            return render_template("recipe.html", plan=None, formula="", batch=5.0, pred=None, error=None)
        try:
            batch = max(0.1, min(float(request.args.get("batch", 5.0)), 1000.0))
        except ValueError:
            batch = 5.0
        try:
            plan = synthesis_plan(f, batch)
            pred = formula_estimate(f + (" " + request.args["sg"] if request.args.get("sg") else ""),
                                    base=request.args.get("base", "").strip() or None)
        except FormulaError as exc:
            return render_template("recipe.html", plan=None, formula=f, batch=batch, pred=None, error=str(exc)), 400
        return render_template("recipe.html", plan=plan, formula=f, batch=batch, pred=pred, error=None, sg=request.args.get("sg", ""))

    @app.route("/api/recipe")
    def api_recipe():
        try:
            return jsonify(synthesis_plan(request.args.get("formula", ""), float(request.args.get("batch", 1.0))))
        except (FormulaError, ValueError) as exc:
            return jsonify({"error": str(exc)}), 400

    # ------------------------------------------------------------------ errors
    @app.errorhandler(404)
    def nf(_):
        return render_template("message.html", title="Not found",
                               text="That page or material is not in this CrystalScout install."), 404

    @app.errorhandler(400)
    def bad(e):
        return render_template("message.html", title="Request not accepted",
                               text=getattr(e, "description", "") or "The request could not be processed."), 400

    @app.errorhandler(403)
    def forbidden(_):
        return render_template("message.html", title="Not allowed",
                               text="Your account role doesn't have access to that page."), 403

    @app.errorhandler(413)
    def too_big(_):
        return render_template("message.html", title="File too large", text="Uploads are limited to 8 MB."), 413

    return app


app = create_app()

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print("=" * 64)
    print(f"  CrystalScout is running:  http://127.0.0.1:{port}")
    print("  Account passwords are supplied through environment variables:")
    for u, a in ACCOUNTS.items():
        print(f"    CRYSTALSCOUT_PW_{u.upper():<10s} {ROLES[a['role']]['label']}")
    print("=" * 64)
    print("Loading models and the crystal-structure library (first start can take a minute) ...")
    STATE.ensure()
    STATE.default_run()
    from crystalscout.ml.structgen import _existence_index, templates

    print(f"  {len(templates())} crystal templates, {len(_existence_index())} known compositions indexed.")
    print(f"Ready. Open http://127.0.0.1:{port} in your browser.")
    app.run(host="127.0.0.1", port=port, debug=False, threaded=True)
