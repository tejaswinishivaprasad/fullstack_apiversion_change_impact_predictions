# run_deterministic.py
# deterministic risk scoring with percentile based bands
import pandas as pd
import numpy as np
import os

IN = "analysis_outputs/features.csv"
OUT = "analysis_outputs/deterministic_predictions.csv"

os.makedirs(os.path.dirname(OUT), exist_ok=True)

df = pd.read_csv(IN)

def base_score(r):
    # hard positive rules
    if int(r.get("breaking_change", 0)) == 1:
        return 1.0
    if int(r.get("is_remove", 0)) == 1:
        return 1.0

    # features used to build a continuous score
    shared = float(r.get("shared_schemas_count", 0))
    calls = float(r.get("calls_services_count", 0))
    req = 1.0 if int(r.get("has_request_body", 0)) == 1 else 0.0
    resp = 1.0 if int(r.get("has_response_body", 0)) == 1 else 0.0
    path_depth = float(r.get("path_depth", 0))

    # normalize counts to small range
    s_shared = min(shared / 5.0, 1.0)
    s_calls = min(calls / 5.0, 1.0)
    s_req = req * 0.25
    s_resp = resp * 0.20
    s_path = min(path_depth / 10.0, 1.0) * 0.15

    # weighted sum
    score = 0.40 * s_shared + 0.30 * s_calls + 0.20 * (s_req + s_resp) + 0.10 * s_path
    # clip
    return float(max(0.0, min(1.0, score)))

# compute base scores for whole df
df["_base"] = df.apply(base_score, axis=1)

# ensure breaking removals are exactly 1.0
mask_hard = (df.get("breaking_change", 0).fillna(0).astype(int) == 1) | (df.get("is_remove", 0).fillna(0).astype(int) == 1)
df.loc[mask_hard, "_base"] = 1.0

# map to bands using percentiles so we get low medium high spread
p90 = df["_base"].quantile(0.90)
p50 = df["_base"].quantile(0.50)

def band_from_score(s):
    if s >= p90:
        return "high"
    if s >= p50:
        return "medium"
    return "low"

df["risk_score"] = df["_base"].round(3)
df["band"] = df["risk_score"].apply(band_from_score)

out = df.loc[:, ["ace_id", "risk_score", "band"]]
out.to_csv(OUT, index=False)
print("Written", OUT)
print("counts by band")
print(out["band"].value_counts().to_string())
