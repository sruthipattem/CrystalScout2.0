"""CrystalScout REST API built with FastAPI.

Start it from the project folder:

    python -m uvicorn api.main:app --port 8000

Then open http://127.0.0.1:8000/docs to try every endpoint in the browser.
It uses the same trained models, data and four accounts as the Flask web app.
"""
from __future__ import annotations

from typing import Dict, List, Optional

from fastapi import Depends, FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel, Field

from . import security, service

app = FastAPI(
    title="CrystalScout API",
    version="2.1",
    description="AI screening of lithium solid electrolytes. Sign in with POST /auth/login, "
                "click **Authorize** and paste the token, then try the endpoints.",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5000", "http://127.0.0.1:5000", "http://localhost:5173",
                   "http://127.0.0.1:5173", "http://localhost:3000", "http://127.0.0.1:3000"],
    allow_methods=["*"],
    allow_headers=["*"],
)
_bearer = HTTPBearer(auto_error=False)


# ------------------------------------------------------------------ request bodies
class LoginBody(BaseModel):
    username: str = Field(..., examples=["scientist"])
    password: str = Field(..., examples=["your scientist password"])


class FormulaBody(BaseModel):
    formula: str = Field(..., examples=["Li6PS5Cl"])
    space_group: Optional[str] = Field(None, examples=["F-43m"], description="optional Hermann-Mauguin symbol")


class DescriptorBody(BaseModel):
    LLB: float = Field(..., description="Li-Li bond number")
    SBI: float = Field(..., description="Sublattice bond ionicity")
    AFC: float = Field(..., description="Anion framework coordination")
    LASD: float = Field(..., description="Li-anion separation distance (Å)")
    LLSD: float = Field(..., description="Li-Li separation distance (Å)")


class ScreenBody(BaseModel):
    band_gap_min: float = Field(1.0, description="keep materials with band gap above this (eV)")
    ehull_max: float = Field(0.0, description="max energy above hull (eV/atom); 0 = stable only")
    exclude_transition_metals: bool = True
    exclude_elements: List[str] = Field(default_factory=list, examples=[["Pb", "Cd"]])
    max_nsites: int = Field(0, description="largest unit cell allowed; 0 = no limit")
    rank_model: str = Field("crystalscout", description="crystalscout, sigma, logreg, random_forest, gboost or sendek2019")
    top: int = Field(20, description="how many ranked results to return (1-500)")
    source: str = Field("snapshot", description="'snapshot' (offline) or 'live' Materials Project")
    mp_api_key: Optional[str] = Field(None, description="needed only for source='live'")


# ------------------------------------------------------------------ helpers
def current_user(creds: Optional[HTTPAuthorizationCredentials] = Depends(_bearer)) -> str:
    try:
        return security.verify(creds.credentials if creds else None)
    except security.LoginError as exc:
        raise HTTPException(status_code=401, detail=str(exc), headers={"WWW-Authenticate": "Bearer"})


def _call(fn, *args, **kwargs):
    try:
        return fn(*args, **kwargs)
    except service.InputError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except service.NotFound as exc:
        raise HTTPException(status_code=404, detail=str(exc))


# ------------------------------------------------------------------ routes
@app.get("/health", tags=["status"])
def health() -> Dict:
    """Is the API up, and which model is ranking? (no sign-in needed)"""
    return service.health()


@app.post("/auth/login", tags=["sign-in"])
def login(body: LoginBody) -> Dict:
    """Exchange the CrystalScout username and password for a token."""
    try:
        return security.login(body.username, body.password)
    except security.LoginError as exc:
        raise HTTPException(status_code=429 if exc.locked else 401, detail=str(exc))


@app.get("/auth/me", tags=["sign-in"])
def me(user: str = Depends(current_user)) -> Dict:
    info = security.user_info(user)
    return {"username": user, "full_name": info.get("full_name"), "role": info.get("role"),
            "role_label": info.get("role_label")}


@app.get("/models", tags=["models"])
def models(user: str = Depends(current_user)) -> Dict:
    """Every model with its held-out test metrics."""
    return service.models_info()


@app.get("/features", tags=["models"])
def features(user: str = Depends(current_user)) -> Dict:
    """What each of the 5 descriptors means."""
    return service.feature_info()


@app.post("/predict/cif", tags=["predict"])
async def predict_cif(file: UploadFile = File(..., description="a .cif crystal structure file"),
                      user: str = Depends(current_user)) -> Dict:
    """Upload a CIF and get its score, verdict and explanation."""
    raw = await file.read()
    if len(raw) > service.MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=400, detail="the CIF file is larger than 8 MB")
    return _call(service.predict_cif, raw.decode("utf-8", errors="replace"), name=file.filename or "upload.cif")


@app.post("/predict/formula", tags=["predict"])
def predict_formula(body: FormulaBody, user: str = Depends(current_user)) -> Dict:
    """Predicted room-temperature conductivity from a chemical formula (no structure needed)."""
    return _call(service.predict_formula, body.formula, body.space_group)


class AdviseBody(BaseModel):
    formula: str = Field(..., examples=["Li6PS5Cl"])
    space_group: Optional[str] = Field(None, examples=["F-43m"])
    top: int = Field(10, ge=1, le=100)
    allow_transition_metals: bool = False
    avoid_toxic: bool = True
    avoid_supply_risk: bool = False


class LabBody(BaseModel):
    formula: str = Field(..., examples=["Li6PS5Cl0.5Br0.5"])
    space_group: Optional[str] = Field(None, examples=["F-43m"])
    baseline: Optional[str] = Field(None, examples=["Li6PS5Cl"])


class RecipeBody(BaseModel):
    formula: str = Field(..., examples=["Li6PS5Cl0.5Br0.5"])
    batch_g: float = Field(1.0, examples=[5.0])


@app.post("/design/lab", tags=["design"])
def design_lab(body: LabBody, user: str = Depends(current_user)) -> Dict:
    """Formula Lab: predict an edited formula and explain the change against a baseline."""
    return _call(service.design_lab, body.formula, body.space_group, body.baseline)


@app.post("/design/advise", tags=["design"])
def design_advise(body: AdviseBody, user: str = Depends(current_user)) -> Dict:
    """Dopant Advisor: the best charge-balanced doped variants of a base compound."""
    return _call(service.design_advise, body.formula, body.space_group, body.top, body.allow_transition_metals,
                 body.avoid_toxic, body.avoid_supply_risk)


class CheckBody(BaseModel):
    formula: str = Field(..., examples=["Li6PS5Cl0.3Br0.7"])
    space_group: Optional[str] = Field(None, examples=["F-43m"])
    baseline: Optional[str] = Field(None, examples=["Li6PS5Cl"], description="parent compound to build the crystal on")


@app.post("/design/check", tags=["design"])
def design_check(body: CheckBody, user: str = Depends(current_user)) -> Dict:
    """Reality check: does the formula exist (Materials Project, OBELiX), is it feasible, its crystal and its prediction."""
    return _call(service.design_check, body.formula, body.space_group, body.baseline)


@app.post("/design/cif", tags=["design"])
def design_cif(body: CheckBody, user: str = Depends(current_user)) -> Dict:
    """A CIF for any formula: the real structure when known, else a site-substitution model (unrelaxed)."""
    return _call(service.design_cif, body.formula, body.space_group, body.baseline)


@app.post("/design/recipe", tags=["design"])
def design_recipe(body: RecipeBody, user: str = Depends(current_user)) -> Dict:
    """Synthesis Planner: precursors, molar ratio and grams for a batch."""
    return _call(service.design_recipe, body.formula, body.batch_g)


@app.post("/predict/descriptors", tags=["predict"])
def predict_descriptors(body: DescriptorBody, user: str = Depends(current_user)) -> Dict:
    """Score a material directly from its 5 descriptors."""
    return _call(service.predict_descriptors, body.model_dump())


@app.post("/screen", tags=["screen"])
def screen(body: ScreenBody, user: str = Depends(current_user)) -> Dict:
    """Run the Stage-A filters, score the survivors and return the ranked shortlist."""
    if body.source == "live" and not security.user_info(user).get("can_live"):
        raise HTTPException(status_code=403, detail="expo guest accounts can't query the live Materials Project")
    return _call(service.screen, body.band_gap_min, body.ehull_max, body.exclude_transition_metals,
                 body.exclude_elements, body.max_nsites, body.rank_model, body.top, body.source,
                 body.mp_api_key, run_by=user)


@app.get("/runs", tags=["screen"])
def runs(user: str = Depends(current_user)) -> List[Dict]:
    """All saved screening runs, newest first."""
    return service.runs()


@app.get("/runs/{run_id}", tags=["screen"])
def run_detail(run_id: str, top: int = Query(20, ge=1, le=500), user: str = Depends(current_user)) -> Dict:
    """One saved run: its summary and top ranked materials."""
    return _call(service.run_detail, run_id, top)
