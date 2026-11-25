"""Command line interface.

    python -m crystalscout.cli build          # descriptors for OBELiX + MP pool, train models
    python -m crystalscout.cli screen         # Stage A on the bundled snapshot -> ranked shortlist
    python -m crystalscout.cli screen --live  # same against the live Materials Project (MP_API_KEY)
    python -m crystalscout.cli predict a.cif b.cif
    python -m crystalscout.cli evaluate       # print validation report
    python -m crystalscout.cli serve          # start the web app
"""
from __future__ import annotations

import argparse
import json
import sys

import pandas as pd


def _bundle():
    from .ml.models import get_bundle

    return get_bundle()


def cmd_build(a):
    from .data.obelix import build_features, training_table
    from .featurize import load_pool_features
    from .ml.models import save_bundle, train_all

    print("1/3  OBELiX descriptors ...")
    f = build_features(use_cache=False, verbose=True)
    print(f"     {(f.excluded_reason == '').sum()} of {len(f)} CIFs usable")
    print("2/3  Materials Project pool descriptors ...")
    p = load_pool_features(rebuild=True)
    print(f"     {len(p)} structures")
    print("3/3  training and validating models ...")
    b = train_all(training_table(), verbose=True)
    save_bundle(b)
    _print_report(b.report)


def _print_report(rep):
    print(f"\n{'model':44s} {'test AUC':>9s} {'LOFO AUC':>9s} {'LOFO F1':>8s} {'enrich':>7s}")
    for k, m in rep["models"].items():
        print(f"{m['label']:44s} {m['test']['roc_auc']:9.3f} {m['cv_all_metrics']['roc_auc']:9.3f} "
              f"{m['cv_all_metrics']['f1']:8.2f} {m['cv_all_metrics']['enrichment']:6.1f}x")


def cmd_evaluate(a):
    from .ml.benchmark import run_benchmark

    b = _bundle()
    _print_report(b.report)
    bm = run_benchmark(b)
    print(f"\nDFT-MD benchmark (Sendek 2019 Tables 1-2): {bm['n']} materials, {bm['n_pos']} superionic")
    for k, m in bm["models"].items():
        print(f"  {m['label']:44s} AUC {m['roc_auc']:.3f}  top-{m['k']} hits {m['hits_in_top_k']}")


def cmd_screen(a):
    from .data.stage_a import StageACriteria
    from .screening import run_screening

    c = StageACriteria(band_gap_min=a.band_gap, ehull_max=a.ehull, exclude_transition_metals=not a.allow_tm,
                       exclude_elements=tuple(a.exclude or ()))

    def prog(stage, i, n):
        print(f"\r  {stage}: {i}/{n}", end="", file=sys.stderr)

    run = run_screening(_bundle(), c, source="live" if a.live else "snapshot", api_key=a.api_key,
                        rank_model=a.rank_model, progress=prog)
    print(file=sys.stderr)
    s = run.summary()
    print("Funnel: " + " -> ".join(f"{lab} {n}" for lab, n in s["funnel"]))
    cols = ["rank", "material_id", "formula", "band_gap", "p_logreg", "p_random_forest", "p_gboost", "p_sendek2019",
            "in_domain", "top_driver"]
    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print(run.shortlist[cols].head(a.top).round(3).to_string(index=False))
    print(f"\nSaved to data/runs/{run.run_id}/ (ranked_candidates.csv, stage_a_rejections.csv.gz, run.json)")
    if a.pdf:
        from .report import shortlist_pdf

        with open(a.pdf, "wb") as fh:
            fh.write(shortlist_pdf(run, _bundle().report, top_n=a.top))
        print(f"PDF report: {a.pdf}")


def cmd_predict(a):
    from .core.descriptors import FEATURES
    from .featurize import featurize_cif_text
    from .ml.explain import applicability, explain_sentence, logreg_contributions
    import numpy as np

    b = _bundle()
    rows = []
    for path in a.cifs:
        try:
            st, f = featurize_cif_text(open(path, encoding="utf-8", errors="replace").read(), name=path)
            x = np.array([f[k] for k in FEATURES])
            e = logreg_contributions(b, x)
            rows.append({"file": path, "formula": st.formula(), **{k: f[k] for k in FEATURES},
                         **{f"p_{m}": round(float(b.models[m].predict_proba(x[None])[0, 1]), 3)
                            for m in ("logreg", "random_forest", "gboost", "sendek2019")},
                         "in_domain": applicability(b, x)["in_domain"], "why": explain_sentence(e)})
        except Exception as exc:  # noqa: BLE001
            rows.append({"file": path, "error": str(exc)})
    df = pd.DataFrame(rows)
    if a.out:
        df.to_csv(a.out, index=False)
        print(f"wrote {a.out}")
    with pd.option_context("display.width", 220, "display.max_columns", 30, "display.max_colwidth", 70):
        print(df.to_string(index=False))


def cmd_validate_advisor(a):
    from .ml.validation import run_all

    print("Family hold-out validation of the sigma-Regressor and the Dopant Advisor ...")
    run_all(save=True, verbose=True)
    print("Saved data/models/advisor_validation.json (shown on the Model validation page).")


def cmd_serve(a):
    from webapp.app import app

    print(f"CrystalScout running at http://127.0.0.1:{a.port}")
    print("Account passwords must be configured with CRYSTALSCOUT_PW_<USERNAME> environment variables.")
    app.run(host=a.host, port=a.port, threaded=True)


def main(argv=None):
    p = argparse.ArgumentParser(prog="crystalscout", description="Screen Li solid electrolytes (Sendek method, OBELiX-trained).")
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("build", help="recompute descriptors and retrain models").set_defaults(fn=cmd_build)
    sub.add_parser("evaluate", help="print validation results").set_defaults(fn=cmd_evaluate)
    s = sub.add_parser("screen", help="run Stage A + ranking")
    s.add_argument("--live", action="store_true", help="query the live Materials Project API")
    s.add_argument("--api-key", default=None, help="Materials Project API key (default: $MP_API_KEY)")
    s.add_argument("--band-gap", type=float, default=1.0)
    s.add_argument("--ehull", type=float, default=0.0)
    s.add_argument("--allow-tm", action="store_true", help="keep transition-metal compounds")
    s.add_argument("--exclude", nargs="*", help="extra elements to exclude")
    s.add_argument("--rank-model", default="logreg", choices=["logreg", "random_forest", "gboost", "sendek2019"])
    s.add_argument("--top", type=int, default=20)
    s.add_argument("--pdf", default=None, help="also write a PDF shortlist report to this path")
    s.set_defaults(fn=cmd_screen)
    pr = sub.add_parser("predict", help="score CIF files")
    pr.add_argument("cifs", nargs="+")
    pr.add_argument("--out", default=None, help="write results CSV")
    pr.set_defaults(fn=cmd_predict)
    sub.add_parser("validate-advisor", help="family hold-out test of the design tools").set_defaults(fn=cmd_validate_advisor)
    sv = sub.add_parser("serve", help="start the web app")
    sv.add_argument("--host", default="127.0.0.1")
    sv.add_argument("--port", type=int, default=5000)
    sv.set_defaults(fn=cmd_serve)
    a = p.parse_args(argv)
    a.fn(a)


if __name__ == "__main__":
    main()
