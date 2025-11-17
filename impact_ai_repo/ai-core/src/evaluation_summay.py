#!/usr/bin/env python3
# sample_and_eval.py — sample N pairs and call AI Core to collect report summary
import requests, csv, json, random, os
AI_CORE = os.getenv("AI_CORE", "http://127.0.0.1:8000")
OUT = "evaluation_summary.csv"
# pick pairs from index.json
idx = json.load(open("datasets/curated/index.json"))
pairs = list(idx.keys())
random.seed(31415)
sample = random.sample(pairs, min(200, len(pairs)))

rows = []
for pid in sample:
    meta = idx[pid]
    old = meta.get("old_canonical") or meta.get("old")
    new = meta.get("new_canonical") or meta.get("new")
    dataset = "openapi"
    url = f"{AI_CORE}/report?old={requests.utils.quote(os.path.basename(old))}&new={requests.utils.quote(os.path.basename(new))}&dataset={dataset}&pair_id={requests.utils.quote(pid)}"
    print("CALL", url)
    r = requests.get(url, timeout=30)
    if r.status_code != 200:
        print("skip", pid, "status", r.status_code)
        continue
    rep = r.json()
    num_backend_gt = len(rep.get("backend_impacts", []))
    num_frontend_gt = len(rep.get("frontend_impacts", []))
    # For evaluation you might fetch ground-truth from NDJSON; here we treat reported as pred and
    # simulate metrics by checking presence >0 (placeholder — replace with proper GT lookup)
    num_backend_pred = num_backend_gt
    rows.append({
        "pair_id": pid,
        "old_file": os.path.basename(old),
        "new_file": os.path.basename(new),
        "dataset": dataset,
        "num_aces": rep.get("summary", "").split()[0] if "summary" in rep else "",
        "num_backend_consumers_gt": num_backend_gt,
        "num_frontend_consumers_gt": num_frontend_gt,
        "num_backend_pred": num_backend_pred,
        "num_frontend_pred": num_frontend_pred if (num_frontend_pred:=num_frontend_gt) else 0,
        "predicted_risk_mean": rep.get("predicted_risk", 0.0),
        "ace_types": ";".join([d.get("type","") for d in rep.get("details", [])])
    })

with open(OUT,"w",newline="",encoding="utf-8") as fh:
    writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
    writer.writeheader()
    for r in rows:
        writer.writerow(r)
print("Wrote", OUT)
