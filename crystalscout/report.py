"""PDF shortlist report (hand-off document for DFT-MD / lab validation)."""
from __future__ import annotations

import io
from typing import Dict

import pandas as pd
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

from .core.descriptors import FEATURES

INK = colors.HexColor("#17202B")
MUTED = colors.HexColor("#5B6672")
BRAND = colors.HexColor("#5B3FB0")
LINE = colors.HexColor("#D9DEE2")


def shortlist_pdf(run, report: Dict, top_n: int = 25) -> bytes:
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=landscape(A4), leftMargin=14 * mm, rightMargin=14 * mm,
                            topMargin=14 * mm, bottomMargin=14 * mm, title="CrystalScout shortlist",
                            author="CrystalScout 2.0")
    ss = getSampleStyleSheet()
    h1 = ParagraphStyle("h1", parent=ss["Title"], fontName="Helvetica-Bold", fontSize=20, textColor=INK, alignment=0, spaceAfter=4)
    h2 = ParagraphStyle("h2", parent=ss["Heading2"], fontName="Helvetica-Bold", fontSize=12.5, textColor=INK, spaceBefore=10)
    body = ParagraphStyle("b", parent=ss["BodyText"], fontName="Helvetica", fontSize=9.2, leading=13, textColor=INK)
    small = ParagraphStyle("s", parent=body, fontSize=7.8, leading=10.5, textColor=MUTED)
    s = run.summary()
    lr = report["models"]["logreg"]
    try:
        from .ml.regression import REPORT_FILE
        import json as _json
        reg_mae = f"{_json.loads(REPORT_FILE.read_text())['models']['extra_trees']['test']['mae']:.2f}"
    except Exception:  # noqa: BLE001
        reg_mae = "n/a"
    from .ml.insight import add_industry_columns
    story = [
        Paragraph("CrystalScout shortlist: candidate Li solid electrolytes", h1),
        Paragraph(f"Run {s['run_id']} · {s['created']} · source: {'live Materials Project API' if s['source'] == 'live' else 'bundled Materials Project snapshot (2026-07-20)'}", small),
        Spacer(1, 6),
        Paragraph("Stage A criteria: " + "; ".join(s["criteria_text"]) + ".", body),
        Paragraph("Funnel: " + " → ".join(f"{lab} {n:,}" for lab, n in s["funnel"]) + f" → {s['n_scored']:,} scored.", body),
        Paragraph(
            "Default ranking: the CrystalScout Score, the equal-weight average of two percentile ranks: the "
            "σ-Regressor (Extra Trees on composition + structure, trained on measured OBELiX conductivities, "
            f"held-out test MAE {reg_mae} log units) and the structure model below. Predicted σ is shown with "
            "its 80 % interval.", body),
        Paragraph(
            f"Structure model: logistic regression on 5 structural descriptors, trained on {report['protocol']['n_train']} OBELiX "
            f"structures with measured RT conductivity. Leave-family-out CV ROC-AUC {lr['cv_all_metrics']['roc_auc']:.2f}; "
            f"official OBELiX test ROC-AUC {lr['test']['roc_auc']:.2f}. Probabilities are screening priorities, "
            "not measured conductivities; every candidate needs DFT-MD or experimental confirmation.", body),
        Spacer(1, 8),
        Paragraph(f"Top {top_n} candidates", h2),
    ]
    sl = add_industry_columns(run.shortlist.head(top_n))
    head = ["#", "MP ID", "Formula", "Family", "Gap (eV)", "CS Score", "σ pred (S/cm)", "80 % range", "P(LR)",
            "P(RF)", "P(GB)", "Flags", "In domain", "OBELiX σ (S/cm)"]
    rows = [head]
    for _, r in sl.iterrows():
        def f(v, d=2):
            return "" if pd.isna(v) else f"{float(v):.{d}f}"
        def e(v):
            return "" if pd.isna(v) else f"{10 ** float(v):.1e}"
        flags = ", ".join(x for x in [("supply: " + r["supply_risk"]) if r["supply_risk"] else "",
                                      ("tox: " + r["toxic"]) if r["toxic"] else "",
                                      "moisture" if r["moisture_sensitive"] else ""] if x)
        rows.append([int(r["rank"]), r["material_id"], r["formula"], r["family"], f(r["band_gap"]), f(r.get("cs_score")),
                     e(r.get("log10_sigma_pred")), f"{e(r.get('log10_sigma_lo'))} – {e(r.get('log10_sigma_hi'))}",
                     f(r["p_logreg"]), f(r["p_random_forest"]), f(r["p_gboost"]), flags]
                    + ["yes" if r.get("in_domain") in (True, "True") else "check",
                       "" if pd.isna(r.get("obelix_sigma")) else f"{float(r['obelix_sigma']):.1e}"])
    t = Table(rows, repeatRows=1)
    t.setStyle(TableStyle([
        ("FONT", (0, 0), (-1, 0), "Helvetica-Bold", 7.6), ("FONT", (0, 1), (-1, -1), "Helvetica", 7.6),
        ("TEXTCOLOR", (0, 0), (-1, 0), BRAND), ("LINEBELOW", (0, 0), (-1, 0), 0.8, INK),
        ("LINEBELOW", (0, 1), (-1, -1), 0.25, LINE), ("ALIGN", (4, 0), (10, -1), "RIGHT"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"), ("BOTTOMPADDING", (0, 0), (-1, -1), 3), ("TOPPADDING", (0, 0), (-1, -1), 3),
    ]))
    story += [t, PageBreak(), Paragraph("Why each candidate scored high", h2)]
    for _, r in sl.iterrows():
        why = r.get("why") if isinstance(r.get("why"), str) and r.get("why") else \
            "No crystal structure in the offline pool, so it is ranked on predicted σ from composition."
        note = r.get("domain_note") if isinstance(r.get("domain_note"), str) else ""
        sig = "" if pd.isna(r.get("log10_sigma_pred")) else f" Predicted σ ≈ {10 ** float(r['log10_sigma_pred']):.1e} S/cm."
        story.append(Paragraph(f"<b>{int(r['rank'])}. {r['formula']}</b> ({r['material_id']}): {why}{sig}"
                               + (f" <font color='#A0522D'>Note: {note}.</font>" if note else ""), body))
    story += [Spacer(1, 8), Paragraph("Method and limits", h2), Paragraph(
        "Descriptors (LLB, SBI, AFC, LASD, LLSD) follow Sendek et al., Energy Environ. Sci. 2017 and Chem. Mater. 2019, "
        "computed with CrystalScout's documented geometric definitions. The oxidation-potential filter (> 4 V) of the base "
        "paper is not applied (future work). Training labels: OBELiX (Therrien et al., 2025, CC BY 4.0). Materials Project "
        "data CC BY 4.0. 'In domain' flags candidates whose descriptors lie inside the training distribution.", body)]
    doc.build(story)
    return buf.getvalue()
