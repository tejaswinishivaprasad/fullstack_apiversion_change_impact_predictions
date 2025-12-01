// src/services/api.js
// Axios-based API wrapper for the Impact UI.
// Targets the Spring proxy at REACT_APP_BACKEND_URL (default http://127.0.0.1:8080/analysis)

import axios from "axios";

const BACKEND_URL = (process.env.REACT_APP_BACKEND_URL || "http://127.0.0.1:8080/analysis").replace(/\/$/, "");
const DEFAULT_TIMEOUT = 10_000;
const JSON_TYPE_RE = /application\/json|text\/json/;

// ----- utilities -----
const unwrap = (res) => {
  if (!res) return {};
  // axios responses have .data
  const payload = res.data ?? res;
  if (payload && typeof payload === "object") return payload;

  if (typeof payload === "string") {
    const s = payload.trim();
    if (s === "") return payload;
    try {
      return JSON.parse(s);
    } catch {
      // sometimes backends double-encode JSON strings; try again
      try {
        const once = JSON.parse(s);
        if (typeof once === "string") return JSON.parse(once);
        return once;
      } catch {
        return s;
      }
    }
  }
  return payload;
};

async function fetchWithRetry(url, opts = {}, retries = 1, timeout = DEFAULT_TIMEOUT) {
  let lastErr = null;
  for (let attempt = 0; attempt <= retries; attempt++) {
    try {
      const source = axios.CancelToken.source();
      const timer = setTimeout(() => source.cancel(`timeout ${timeout}ms`), timeout);
      const cfg = {
        url,
        timeout,
        cancelToken: source.token,
        ...opts,
      };
      const res = await axios(cfg);
      clearTimeout(timer);
      return res;
    } catch (err) {
      lastErr = err;
      // simple backoff
      await new Promise((r) => setTimeout(r, 200 * (attempt + 1)));
    }
  }
  throw lastErr;
}

function buildQuery(params = {}) {
  const kv = [];
  for (const [k, v] of Object.entries(params)) {
    if (v === undefined || v === null) continue;
    kv.push(`${encodeURIComponent(k)}=${encodeURIComponent(v)}`);
  }
  return kv.length ? `?${kv.join("&")}` : "";
}

// ----- endpoints -----

export const fetchDatasets = async () => {
  const url = `${BACKEND_URL}/datasets`;
  const res = await fetchWithRetry(url, { method: "GET", headers: { Accept: "application/json" } }, 1);
  return unwrap(res);
};

export const fetchFiles = async (dataset) => {
  if (!dataset) throw new Error("dataset required");
  // controller routes this to /files?dataset=...
  const url = `${BACKEND_URL}/datasets/${encodeURIComponent(dataset)}`;
  const res = await fetchWithRetry(url, { method: "GET", headers: { Accept: "application/json" } }, 1);
  return unwrap(res);
};

export const analyzeAPI = async (oldFile, newFile, dataset = "openapi", pairId = null) => {
  if (!oldFile || !newFile) throw new Error("oldFile and newFile required");
  // Spring expects /analysis/report which proxies to AI core /report
  const params = {
    old: oldFile,
    new: newFile,
    dataset,
  };
  if (pairId) params.pair_id = pairId; // note snake_case
  const url = `${BACKEND_URL}/report${buildQuery(params)}`;
  const res = await fetchWithRetry(url, { method: "GET", headers: { Accept: "application/json" } }, 1, 12_000);
  return unwrap(res);
};

export const fetchConsumers = async (service, path = null) => {
  if (!service) throw new Error("service required");
  const params = { service };
  if (path) params.path = path;
  const url = `${BACKEND_URL}/consumers${buildQuery(params)}`;
  const res = await fetchWithRetry(url, { method: "GET", headers: { Accept: "application/json" } }, 1);
  return unwrap(res);
};

export const fetchVersioning = async (pairId) => {
  if (!pairId) throw new Error("pairId required");
  // Spring forwards /analysis/versioning?pair_id=... to AI core /versioning
  const url = `${BACKEND_URL}/versioning${buildQuery({ pair_id: pairId })}`;
  const res = await fetchWithRetry(url, { method: "GET", headers: { Accept: "application/json" } }, 1);
  return unwrap(res);
};

export const fetchGraph = async () => {
  const url = `${BACKEND_URL}/graph`;
  const res = await fetchWithRetry(url, { method: "GET", headers: { Accept: "application/json" } }, 1);
  return unwrap(res);
};

export const trainModel = async (samples) => {
  const url = `${BACKEND_URL}/train`;
  const res = await fetchWithRetry(url, {
    method: "POST",
    headers: { "Content-Type": "application/json", Accept: "application/json" },
    data: samples,
  }, 1, 15_000);
  return unwrap(res);
};

// Batch reports (unchanged logic, just using the above helpers)
export async function fetchReportsBatch({ dataset, limit = 10, onProgress = null }) {
  const filesResp = await fetchFiles(dataset);
  const files = (filesResp && filesResp.samples) || [];

  const v1Candidates = files.filter((f) => /-v1\b/i.test(f) || /\bv1\b/i.test(f));
  const pairs = [];
  for (let f1 of v1Candidates) {
    const base = f1.replace(/-v1/i, "").replace(/\bv1\b/i, "");
    const f2 =
      files.find((x) => x !== f1 && x.includes(base) && /-v2\b/i.test(x)) ||
      files.find((x) => x !== f1 && x.includes(base) && /\bv2\b/i.test(x));
    if (f2) {
      pairs.push({ oldFile: f1, newFile: f2 });
    }
    if (pairs.length >= limit) break;
  }

  if (pairs.length === 0) {
    for (let i = 0; i + 1 < files.length && pairs.length < limit; i += 2) {
      pairs.push({ oldFile: files[i], newFile: files[i + 1] || files[i] });
    }
  }

  const results = [];
  for (let i = 0; i < pairs.length; i++) {
    const p = pairs[i];
    try {
      if (onProgress) onProgress(i + 1, pairs.length, p);
      const report = await analyzeAPI(p.oldFile, p.newFile, dataset, null);
      results.push({ meta: p, report });
    } catch (err) {
      results.push({ meta: p, error: err?.message || String(err) });
    }
  }
  return results;
}

// --- fetchAce: prefer export/ace?inline=true, fallback to /ace ---
export async function fetchAce(pairId, aceId) {
  if (!aceId) throw new Error("aceId required for fetchAce");

  const paramsAce = pairId ? { pair_id: pairId, ace_id: aceId } : { ace_id: aceId };
  const paramsInline = pairId ? { pair_id: pairId, ace_id: aceId, inline: "true" }
                             : { ace_id: aceId, inline: "true" };

  const candidates = [
    `${BACKEND_URL}/ace${buildQuery(paramsAce)}`,                       
    `${BACKEND_URL}/export/ace${buildQuery(paramsInline)}`             
  ];

  let lastErr = null;
  for (const url of candidates) {
    try {
      const res = await fetchWithRetry(url, { method: "GET", headers: { Accept: "application/json, text/*" } }, 0);
      const body = unwrap(res);

      if (body && typeof body === "object") {
        if (body.ace) return body;
        if (body.ace_id || body.aceId || body.type) return { ace: body, source: url };
        return { ace: body, source: url };
      }

      if (typeof body === "string") {
        try {
          const parsed = JSON.parse(body);
          if (parsed) {
            if (parsed.ace) return parsed;
            if (parsed.ace_id || parsed.aceId || parsed.type) return { ace: parsed, source: url };
            return { ace: parsed, source: url };
          }
        } catch (_) {
          return { ace: body, source: url };
        }
      }

      return { ace: body, source: url };
    } catch (err) {
      lastErr = err;
    }
  }

  const msg = lastErr?.message || JSON.stringify(lastErr) || "fetchAce failed: no candidate endpoints succeeded";
  throw new Error(msg);
}





