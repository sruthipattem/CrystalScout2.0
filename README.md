# CrystalScout

**Screen, predict and design lithium solid-state electrolytes, with a reality check on every formula.**
Team Crystal Scout · Group G1544 · Neil Gogte Institute of Technology · Mentor: Dr M. Deepika

CrystalScout extends the method of Sendek et al. (*Chem. Mater.* 2019, 31, 342–352) from a yes/no classifier into a complete discovery platform:

| | What it does |
|---|---|
| **Screen** | Filters 19,806 Materials Project Li compounds to 505 stable, insulating, transition-metal-free candidates and ranks them |
| **Predict** | σ-Regressor (Extra Trees, 500 trees, 96 features) predicts room-temperature ionic conductivity with an 80 % range; a logistic regression on the 5 Sendek descriptors gives P(superionic); both combine into the CrystalScout Score |
| **Check** (new) | Every formula is looked up in Materials Project (19,806 Li compounds, DFT E<sub>hull</sub>) and OBELiX (599 measured), checked for charge balance and ion-size fit, and gets a feasibility badge: *Measured*, *Known · DFT-stable*, *Known · metastable*, *Hypothetical · plausible*, *Hypothetical · strained* or *Unlikely to exist* |
| **Build** (new) | Every formula gets a crystal structure and a downloadable **CIF**: the real one when it exists, otherwise a site-substitution model on the best of 1,578 real host crystals (fractional occupancies, Li re-filled, Vegard-scaled, labelled unrelaxed). The CIF is read back through the structure path, so new compounds are scored with real Sendek descriptors and both models |
| **Design** | **Formula Lab** (live edits, 3D crystal, reality check), **Dopant Advisor** (hard charge and ion-size filters, known compounds kept apart as benchmarks, top 20 new variants scored on generated crystals), **Synthesis Planner** (precursors, molar ratio, grams) |
| **Deliver** | Secure 4-role web app with image captcha, PDF/CSV reports, and a token-secured REST API (`/design/check`, `/design/cif`, `/design/lab`, `/design/advise`, `/design/recipe`) |

Training data: OBELiX, 562 measured conductivities (≈ 14× the base paper's ≈ 40 materials).

### Does it find materials it has never seen? (family hold-out)

The advisor's famous picks (Ta-doped LLZO, Br-mixed argyrodites) are in OBELiX, so a model trained on all of OBELiX has seen them. We therefore retrain the σ-Regressor with **every** entry of a crystal family removed and rerun the advisor (`python -m crystalscout.cli validate-advisor`):

| Family hidden from training | Advisor run on | Variants that were really made, and where the advisor ranked them |
|---|---|---|
| all 104 garnets | Li₇La₃Zr₂O₁₂ (94 variants) | Li₆.₅La₃Zr₁.₅Ta₀.₅O₁₂ **#1**, Li₆.₈La₃Zr₁.₈Ta₀.₂O₁₂ **#2**, Li₆La₃ZrTaO₁₂ **#5** (all measured superionic) |
| all 52 argyrodites | Li₆PS₅Cl (39 variants) | Li₆PS₅Cl₀.₅Br₀.₅ **#2**, Li₆PS₅Cl₀.₇₅Br₀.₂₅ **#3** (both measured > 2 mS/cm) |

Ranking inside an unseen family (Spearman ρ): garnet 0.71, NASICON 0.63, thio-LISICON 0.84, but argyrodite 0.29, LGPS 0.26 and perovskite 0.12. Absolute values are less reliable than the ranking, so every prediction is labelled **a screening priority, not a measured conductivity**.

### Honest limits

- Energy above hull is reported for known compounds only; for hypothetical ones it needs DFT or an ML potential (CHGNet, M3GNet, MACE), which this offline install does not run. Parent stability and ion-size fit are shown as proxies.
- Generated CIFs are unrelaxed, disordered (fractional-occupancy) starting models. Relax and order them before DFT.
- ICSD is licensed; ICSD numbers are shown where OBELiX provides them.

**Accounts**: `admin`, `scientist`, `partner` and `expo`. Set a unique password for each with the `CRYSTALSCOUT_PW_<USERNAME>` environment variables before starting the app; no passwords are bundled with the project.

## 1. Run it (Windows, macOS, Linux)

Needs Python 3.10 or newer.

```bash
cd CrystalScout
python -m pip install -r requirements.txt
```

Before starting the app, configure all four account passwords. In Windows PowerShell:

```powershell
$env:CRYSTALSCOUT_PW_ADMIN = "choose-a-unique-password"
$env:CRYSTALSCOUT_PW_SCIENTIST = "choose-a-unique-password"
$env:CRYSTALSCOUT_PW_PARTNER = "choose-a-unique-password"
$env:CRYSTALSCOUT_PW_EXPO = "choose-a-unique-password"
```

On macOS or Linux, use `export CRYSTALSCOUT_PW_ADMIN='...'` and likewise for `SCIENTIST`, `PARTNER` and `EXPO`. Keep these values private. The app will not start if any account password is missing.

Start the app from the same terminal session where you set the variables:

```bash
python -m webapp.app
```

Open **http://127.0.0.1:5000**. On Windows, `run_app.bat` also works once the required variables are configured in your Windows environment.

### Signing in

The app opens on the sign-in page with an **image CAPTCHA** (type the 5 characters shown; the ↻ button gives a new image). After signing in you land directly on the **Dashboard**. Self-registration is off; four role-based accounts are built in:

| Username | Password configuration | Role | Access |
|---|---|---|---|
| `admin` | `CRYSTALSCOUT_PW_ADMIN` | Administrator | everything + Admin console (accounts, sign-in audit log) |
| `scientist` | `CRYSTALSCOUT_PW_SCIENTIST` | Research Scientist | full screening, scoring, live Materials Project, exports, API |
| `partner` | `CRYSTALSCOUT_PW_PARTNER` | Industry Partner (battery R&D) | same as scientist |
| `expo` | `CRYSTALSCOUT_PW_EXPO` | Expo Guest | everything except live Materials Project and the Admin console |

- Set passwords only through these environment variables; never commit them. Changing a password signs that account out everywhere.
- Passwords are held only as salted hashes and checked in constant time. The CAPTCHA answer is never stored in readable form (only an HMAC), each image allows one attempt and expires after 5 minutes.
- Five wrong passwords lock that username for 10 minutes; every attempt is written to `data/access_log.jsonl`.
- All forms carry CSRF tokens; cookies are HttpOnly and SameSite=Lax.

Everything works offline out of the box. The trained models, OBELiX, and a 19,806-entry Materials Project snapshot are bundled.

### Live Materials Project mode

1. Get a free API key at https://next-gen.materialsproject.org/api.
2. Either paste it into **Run a screen → Live Materials Project**, or set it once:
   - Windows: `set MP_API_KEY=your_key`
   - macOS or Linux: `export MP_API_KEY=your_key`

The live mode works with plain HTTPS. If the official `mp-api` package is installed, CrystalScout uses that instead.

---

## 2. What the web app does

| Page | What you get |
|---|---|
| **Sign in** | Animated light-theme sign-in, 4 role-based accounts, image CAPTCHA, show/hide password, Caps Lock hint, lockout |
| **Dashboard** | Count-up KPIs, the screening funnel, top-8 candidate cards with CrystalScout Score rings and predicted σ, trust metrics, chemistry mix, industry use cases |
| **Quick check** | Type a formula (or drop one CIF) and get predicted σ with its 80 % range, tier, CrystalScout Score and a plain-language verdict |
| **Run a screen** | Industry presets (standard, cost-aware, low-toxicity, high-voltage, broad), Stage A thresholds, snapshot or live Materials Project, ranking choice, background job with progress ring |
| **Candidate shortlist** | Rank by CrystalScout Score, predicted σ or any classifier; filter by chemistry family, supply-risk / toxic elements, training domain; compare up to 4; CSV and PDF report |
| **Material report** | Verdict banner, rotatable 3D crystal, σ gauge with interval, structure-model probabilities, exact explanation, descriptor strips, engineering notes (moisture, supply risk, toxicity), lab value from OBELiX |
| **Compare** | Side-by-side σ ranges chart and every metric for 2–4 candidates |
| **Score materials** | Tabs for CIF upload (drag and drop, up to 50), batch formula estimates, descriptor CSV; CSV export |
| **Developer API** | How to start the REST API, endpoints and a Python example |
| **Formula Lab** | Type or edit a formula with −/+ steppers and element swaps; instant σ, range, tier, charge-balance check, explanation of what changed, and an experiment history |
| **Dopant Advisor** | Base formula in → every charge-balanced doping (10/25/50 %) predicted in one batch → top 10 with gains, flags, measured values when known, and a recipe |
| **Synthesis Planner** | Precursors from automatic binary selection (e.g. Li₂S, P₂S₅, LiCl), NNLS element balance, grams for any batch, route and safety notes, printable lab sheet |
| **Admin console** | Accounts, roles, lockout status and the sign-in audit log (admin only) |
| **Descriptor explorer** | Five sliders that drive the published Sendek equation and the retrained one side by side, with presets for real materials |
| **Model validation** | σ-Regressor metrics, parity plot, feature importance, CrystalScout Score checks, then the classifier protocol, metrics, ROC curves, coefficient comparison, permutation importance, and the DFT-MD benchmark |
| **Training data** | OBELiX statistics, caveat flags, conductivity histogram, and every structure with its descriptors |
| **Methods & scope** | Exact descriptor definitions, scope boundaries, and references |

---

## 3. Command line

```bash
python -m crystalscout.cli screen                     # snapshot screen, prints the top 20, saves data/runs/<id>/
python -m crystalscout.cli screen --live --top 30     # live Materials Project (needs MP_API_KEY)
python -m crystalscout.cli screen --band-gap 2 --exclude Ge --pdf shortlist.pdf
python -m crystalscout.cli predict my1.cif my2.cif --out scores.csv
python -m crystalscout.cli evaluate                   # validation tables
python -m crystalscout.cli build                      # recompute all descriptors and retrain (about 2 minutes)
python -m crystalscout.cli serve --port 5000
```

---

## 4. REST API (FastAPI)

The same models are also available as a JSON API, so other programs (a lab's own tools, a React or mobile front end, a notebook) can use CrystalScout without the web pages.

**Start it:** double-click `run_api.bat` (Windows) or run `bash run_api.sh`, or:

```bash
python -m uvicorn api.main:app --port 8000
```

Open **http://127.0.0.1:8000/docs** for interactive documentation. Every endpoint can be tried there. `run_both.bat` starts the web app (port 5000) and the API (port 8000) together.

**Sign in.** The API uses the same single account as the web app:

1. `POST /auth/login` with `{"username": "scientist", "password": "<your scientist password>"}` returns an `access_token`.
2. Send it on every other request as `Authorization: Bearer <token>`. On the `/docs` page, click **Authorize** and paste it.

Tokens last 12 hours and stop working if the username or password changes. Five wrong passwords lock sign-in for 10 minutes.

| Method | Path | What it does |
|---|---|---|
| GET | `/health` | Status and ranking model (no sign-in needed) |
| POST | `/auth/login` | Username + password → token |
| GET | `/auth/me` | Who the token belongs to |
| GET | `/models` | All four models with their held-out test metrics |
| GET | `/features` | What the five descriptors mean |
| POST | `/predict/cif` | Upload a `.cif` file → descriptors, score, verdict, per-feature explanation |
| POST | `/predict/descriptors` | Send the five descriptors as JSON → score and explanation |
| POST | `/screen` | Stage-A filters + scoring → ranked shortlist (snapshot or live Materials Project) |
| GET | `/runs` | Saved screening runs |
| GET | `/runs/{run_id}` | One saved run with its top results |

**From Python** (standard library only): set `CRYSTALSCOUT_PW_SCIENTIST` in the same terminal session, then with the API running, `python -m api.example_client` signs in, scores a CIF and runs a screen.

**From the command line:**

```bash
curl -X POST http://127.0.0.1:8000/auth/login -H "Content-Type: application/json" \
     -d '{"username":"scientist","password":"<your scientist password>"}'
curl -X POST http://127.0.0.1:8000/predict/cif -H "Authorization: Bearer <token>" \
     -F "file=@data/obelix/cifs/jqc.cif"
```

How it is built: `api/service.py` holds the logic as plain Python functions (calling the same `crystalscout` code as the web app), `api/security.py` issues and checks tokens, and `api/main.py` is a thin FastAPI layer of routes and request schemas.

---

## 5. Results (bundled models)

### σ-Regressor (predicts room-temperature conductivity)

Extra Trees (500 trees) on log₁₀ σ, using Magpie-style composition statistics (11 element properties × 6 statistics, stoichiometry norms, anion fractions), crystal system and centring from the space group, volume per atom, and the five Sendek descriptors when a structure is available. 562 OBELiX materials (upper-bound "<1E-10" values removed), official split 449 / 113.

| Model | Grouped 5-fold CV MAE | CV R² | Test MAE | Test RMSE | Test R² | Superionic AUC (test) |
|---|---|---|---|---|---|---|
| Mean baseline | 1.80 | 0.00 | 1.64 | 2.25 | 0.00 | – |
| Ridge | 1.25 | 0.42 | 1.95 | 2.68 | −0.42 | – |
| Random forest | 0.85 | 0.67 | 1.36 | 1.79 | 0.37 | 0.85 |
| Gradient boosting | 0.85 | 0.68 | 1.29 | 1.74 | 0.40 | – |
| **Extra Trees (σ-Regressor)** | **0.79** | **0.70** | 1.34 | 1.82 | 0.34 | **0.86** |

Every prediction carries an 80 % range from the tree spread, calibrated on out-of-fold residuals (82 % actual coverage on the test split). After evaluation the same model is refit on all 562 rows for use in the app; every number above comes from the train-only model. The team's original Magpie feature file (139 features) was not available offline, so the approach was rebuilt with an equivalent self-contained feature set; the numbers are close to the project overview (CV R² 0.70 vs 0.71).

### CrystalScout Score (default ranking)

Equal-weight average of the σ-Regressor percentile and the structure-model (logistic regression) percentile, weights fixed in advance:

| Independent check (ROC-AUC) | σ-Regressor | Structure model | **CrystalScout Score** |
|---|---|---|---|
| OBELiX test materials with a structure (59) | 0.87 | 0.39 | **0.73** |
| DFT-MD verdicts on MP crystals (38) | 0.63 | 0.77 | **0.73** |

Each ingredient fails one check; the combined score is good on both. In the default shortlist the DFT-MD-confirmed superionic Li₃InCl₆, Li₂B₂S₅, CsLi₂BS₃ and Li₅B₇S₁₃ rank 12, 17, 28 and 51 of 505.

### Structure models (Sendek descriptors)

**Validation.** ROC-AUC: 0.5 = random ordering, 1 = perfect.

| Model | OBELiX test split (67) | Leave-family-out CV (314) | LOFO F1 | Enrichment vs random | DFT-MD benchmark (38) |
|---|---|---|---|---|---|
| **Logistic regression, retrained (ranking model)** | 0.47 | 0.73 | 0.60 | 1.9× | **0.77** |
| Random forest | 0.71 | 0.72 | 0.56 | 1.9× | 0.50 |
| Gradient boosting | 0.57 | 0.64 | 0.45 | 1.3× | 0.57 |
| Sendek 2019 published equation | 0.42 | 0.75 | 0.57 | 2.1× | 0.74* |

\* The published model chose half of the DFT-MD benchmark materials itself, so its score there is biased upward.

How to read this:

- **The retrained equation agrees with the published one.** 4 of 5 coefficients keep Sendek's sign (SBI −, AFC −, LASD +, LLSD −). This is independent confirmation from experimental data.
- **External physics check.** On Sendek's own DFT-MD verdicts for 38 Materials Project materials, the OBELiX-trained logistic regression reaches AUC 0.77 and puts 5 of the 10 superionic materials in its top 10. It never saw any of them.
- **Recovered discoveries.** In the default shortlist, Li₂B₂S₅, CsLi₂BS₃ and Li₅B₇S₁₃ (DFT-MD-confirmed by Sendek et al.) rank 28, 36 and 40 of 505, which is the top 8 %.
- **The OBELiX test split is hard and small.** Twelve near-identical Li₄(Ge,Sn)(S,Se)₄ compounds that look fast by geometry but were measured slow sit together in it. Leave-family-out CV over all 314 structures is the steadier estimate. The ranking model was fixed in advance (the base-paper model family), never chosen by test score.

Scores are screening priorities, not conductivities. A high rank means "promising candidate for DFT-MD or experiment", not "discovered electrolyte".

---

## 6. Project layout

```
crystalscout/
  core/            CIF parser, periodic structure + neighbour search, five descriptors, element data
  data/            OBELiX loader (labels, split, caveat flags), Materials Project snapshot + live client, Stage A
  ml/              model training/validation, exact explanations + applicability domain, DFT-MD benchmark
  featurize.py     one feature extractor shared by Materials Project, OBELiX and uploads
  screening.py     end-to-end run: Stage A → descriptors → scores → ranked shortlist (saved under data/runs/)
  report.py        PDF shortlist report
  cli.py           command line
webapp/            Flask app, sign-in (auth.py), server-rendered SVG charts, canvas 3D viewer (no CDN needed)
api/               FastAPI REST API: main.py (routes), service.py (logic), security.py (tokens), example_client.py
data/
  obelix/          OBELiX all/train/test CSV + 321 CIFs (CC BY 4.0)
  materials_project/  snapshot of 19,806 Li entries + 1,648 structures (CC BY 4.0)
  benchmarks/      Sendek 2019 DFT-MD verdicts (Tables 1–2)
  processed/       cached descriptors
  models/          trained models + evaluation_report.json
docs/METHODS.md    descriptor definitions, protocol and design decisions
docs/DATA.md       data sources, provenance and caveats
scripts/           snapshot rebuild script
tests/             core science, pipeline, web app, sign-in, simple mode, API logic, API routes
```

Run the tests with `python -m pytest -q`.

---

## 7. Scope

**In scope:** live Stage A, OBELiX integration with caveat handling, CIF descriptor extraction (periodic boundaries and partial occupancy), a leakage-free validated logistic regression, model comparison, interpretability, and a ranked shortlist.

**Out of scope, by design:**
- the oxidation-potential > 4 V filter (needs phase-diagram computation; future work)
- new DFT-MD runs
- synthesis
- new ML architectures

## 8. References

- A. D. Sendek et al., *Energy Environ. Sci.* 10, 306 (2017).
- A. D. Sendek et al., *Chem. Mater.* 31, 342 (2019), doi:10.1021/acs.chemmater.8b03272.
- F. Therrien et al., OBELiX, arXiv:2502.14234 (2025). Data CC BY 4.0.
- A. Jain et al., The Materials Project, *APL Mater.* 1, 011002 (2013). Data CC BY 4.0. The offline snapshot comes via the Scandium-Dataset v1.0.0 compilation (MP extraction 2026-07-20).
