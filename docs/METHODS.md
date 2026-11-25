# Methods

## 1. Stage A: property filters

A material passes when it:

- contains Li
- has a band gap > 1 eV
- has E_hull ≤ 0 (numerical tolerance 1e-6 eV/atom)
- contains no transition metals (d-block groups 3–12)

Every failed rule is recorded for every rejected material (`stage_a_rejections.csv.gz`).

The base paper's fourth filter, oxidation potential > 4 V vs Li/Li⁺, needs grand-potential phase diagrams against all competing phases. It is not applied, and is documented as future work.

## 2. Structures

`crystalscout/core/cif.py` reads CIFs:

- loops and quoted values
- symmetry operators under either tag
- values with uncertainties, e.g. `0.1234(5)`
- charged type symbols, e.g. `O2-`
- partial occupancy and mixed sites

Symmetry copies are merged within 0.05 Å. Materials Project structures arrive as pymatgen dictionaries and are read directly. Neighbour search is periodic, over all lattice images within 7 Å.

Validation:

- 278 of 321 OBELiX CIFs reproduce the reported cell composition exactly.
- The other 43 are genuine dataset differences. Most are doped phases described by a parent CIF, and they are flagged.

## 3. Descriptors

| | Definition |
|---|---|
| LLB | Expected number of Li neighbours within 3.0 Å of a Li site, averaged over Li sites |
| SBI | Mean Pauling ionicity 1 − exp(−Δχ²/4) of framework-cation → anion bonds in the first shell (≤ 1.25 × the shortest contact). Li binaries use Li–anion bonds |
| AFC | Mean number of anion neighbours of an anion within the anion sublattice, out to 1.12 × the mean of its 4 shortest anion–anion contacts |
| LASD | Mean Li–anion bond length over each Li's first shell (≤ 1.25 × the shortest Li–anion contact) |
| LLSD | Mean distance from each Li to its nearest Li |

**Anions.** Any of N, O, F, S, Cl, Se, Br, Te, I that is present. If none is present, the most electronegative non-Li element is the anion (H in LiBH₄, P in Li₃P).

**Partial occupancy.** Handled as an expectation. Each site-level quantity is weighted by the element's occupancy on that site, and neighbour counts add the neighbour's occupancy.

**Split sites.** Pairs closer than 1.5 Å (1.6 Å for Li–Li) are alternative positions of one atom and are never counted as neighbours.

**Checks on ideal lattices:**

- Rock-salt LiCl: AFC = 12, LASD = a/2, LLSD = a/√2.
- A bcc anion lattice: AFC = 8.

**Why AFC uses the 4-contact anchor.** A single anomalously short anion–anion contact is common in disordered or perturbed CIFs. With a plain "nearest-distance" shell, one such contact made a garnet's AFC collapse to 1.

## 4. Labels and split

- **Label:** superionic when the room-temperature conductivity is ≥ 1e-4 S/cm. Conductivities reported as "<1E-10" are taken at the bound.
- **Test set:** OBELiX's official split (479 / 120 entries; 247 / 67 with usable CIFs). No composition appears on both sides.
- **Hyperparameters:** chosen by 5-fold CV on the training split, with folds grouped by crystal family.

## 5. Models

| Model | Setup |
|---|---|
| Logistic regression (ranking model) | Standardised inputs, balanced class weights, C from {0.01 … 10} |
| Random forest | 500 trees, balanced, leaf size and depth tuned |
| Gradient boosting | XGBoost if installed, otherwise sklearn HistGradientBoosting |
| Sendek 2019 equation | Fixed published coefficients, as a zero-shot benchmark |

Logistic regression is the ranking model by design: it is the base-paper model family and it is interpretable. Test scores never chose the model.

## 6. Evaluation

1. **OBELiX test split.** Precision, recall, F1, ROC-AUC, PR-AUC, confusion matrix, and a bootstrap 95 % CI for AUC.
2. **Leave-family-out CV over all 314 structures.** 10 folds, StratifiedGroupKFold by family.
3. **External DFT-MD benchmark.** Sendek 2019 Tables 1–2, 38 Materials Project structures with DFT-MD verdicts.

Enrichment is the precision of the top 20 % of picks divided by the base rate. This is the discovery-rate comparison Sendek et al. used ("2.7× random").

## 7. Explanations and trust

**Per-candidate explanation.** Each contribution is θᵢ (xᵢ − μᵢ)/σᵢ in log-odds against the average training material. For a linear model this is exact, and equals SHAP with an interventional training-mean baseline. It describes the model, not physical causation.

**Applicability domain.** A candidate is flagged when any descriptor lies outside the training 1st–99th percentile, or when its Mahalanobis distance exceeds the 97.5th percentile of the training set.

**Experimental cross-check.** When a candidate's reduced formula appears in OBELiX, its measured conductivity is shown next to it.
