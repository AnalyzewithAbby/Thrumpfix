from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import pandas as pd

app = FastAPI(title="Thrumpfix Matching API", version="1.0")

# -------------------------
# 1) CONFIG: update these to match your Excel column headers
# -------------------------
PLUMBER_FILE = "data/plumbers.csv"

COL_PLUMBER_ID = "Plumber ID"   # change if yours differs
COL_LGA        = "LGA"
COL_LCDA       = "LCDA"
COL_REGION     = "Region"

# -------------------------
#2) Load dataset once at startup
# -------------------------
try:
    plumbers_df = pd.read_csv(PLUMBER_FILE)
except Exception as e:
    raise RuntimeError(f"Could not load {PLUMBER_FILE}: {e}")

# Basic validation (fail early with clear error)
required_cols = [COL_PLUMBER_ID, COL_LGA, COL_LCDA, COL_REGION]
missing = [c for c in required_cols if c not in plumbers_df.columns]
if missing:
    raise RuntimeError(f"Missing columns in plumber file: {missing}. Found: {list(plumbers_df.columns)}")

def norm(x) -> str:
    return str(x).strip().lower()

# Precompute normalized columns for faster matching
plumbers_df["_lga_norm"] = plumbers_df[COL_LGA].map(norm)
plumbers_df["_lcda_norm"] = plumbers_df[COL_LCDA].map(norm)
plumbers_df["_region_norm"] = plumbers_df[COL_REGION].map(norm)

# -------------------------
# 3) Request schema
# -------------------------
class MatchRequest(BaseModel):
    jobId: int
    lga: str = Field(..., min_length=1)
    lcda: str = Field(..., min_length=1)
    region: str = Field(..., min_length=1)
    topN: int = Field(3, ge=1, le=20)

@app.get("/health")
def health():
    return {"status": "ok", "version": app.version}

# -------------------------
# 4) The notebook logic -> inside an endpoint
# -------------------------
@app.post("/match")
def match(req: MatchRequest):
    lga_n = norm(req.lga)
    lcda_n = norm(req.lcda)
    region_n = norm(req.region)

    df = plumbers_df.copy()

    # Your scoring logic: LGA > LCDA > Region
    def score_row(row):
        if row["_lga_norm"] == lga_n:
            return 1.0
        elif row["_lcda_norm"] == lcda_n:
            return 0.9
        elif row["_region_norm"] == region_n:
            return 0.7
        else:
            return 0.0

    df["score"] = df.apply(score_row, axis=1)

    # Filter + sort + top N
    result = df[df["score"] > 0].sort_values("score", ascending=False).head(req.topN)

    recommended = []
    for _, r in result.iterrows():
        s = float(r["score"])
        reason = "Same LGA" if s == 1.0 else ("Same LCDA" if s == 0.9 else "Same Region")
        recommended.append({
            "plumberId": int(r[COL_PLUMBER_ID]),
            "score": s,
            "reason": reason
        })

    return {
        "jobId": req.jobId,
        "recommendedPlumbers": recommended
    }