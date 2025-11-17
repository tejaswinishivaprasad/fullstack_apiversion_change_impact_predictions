// src/services/api.js
import axios from "axios";

const BACKEND_URL =
  process.env.REACT_APP_BACKEND_URL || "http://127.0.0.1:8080/analysis";

const DEFAULT_TIMEOUT = 10_000;

// Try to return a usable JS object from various backend response shapes.
const unwrap = (res) => {
  if (!res) return {};
  const payload = res.data ?? res;
  if (payload && typeof payload === "object") return payload;

  if (typeof payload === "string") {
    const s = payload.trim();
    if (s.startsWith("{") || s.startsWith("[")) {
      try {
        return JSON.parse(s);
      } catch {
        // continue to double-parse attempt
      }
    }
    try {
      const once = JSON.parse(s);
      if (typeof once === "string") {
        return JSON.parse(once);
      }
      return once;
    } catch {
      return s;
    }
  }

  return payload;
};

// Simple fetch wrapper with a cancel token and retry/backoff.
async function fetchWithRetry(url, opts = {}, retries = 1, timeout = DEFAULT_TIMEOUT) {
  let lastErr = null;
  for (let attempt = 0; attempt <= retries; attempt++) {
    try {
      const source = axios.CancelToken.source();
      const timer = setTimeout(() => source.cancel(`timeout ${timeout}ms`), timeout);
      const res = await axios({ url, cancelToken: source.token, ...opts });
      clearTimeout(timer);
      return res;
    } catch (err) {
      lastErr = err;
      await new Promise((r) => setTimeout(r, 250 * (attempt + 1)));
    }
  }
  throw lastErr;
}

// Fetch list of datasets from backend.
export const fetchDatasets = async () => {
  const res = await fetchWithRetry(`${BACKEND_URL}/datasets`, {}, 1);
  return unwrap(res);
};

// Fetch files/samples for a dataset.
export const fetchFiles = async (dataset) => {
  const res = await fetchWithRetry(
    `${BACKEND_URL}/datasets/${encodeURIComponent(dataset)}`,
    {},
    1
  );
  return unwrap(res);
};

// Call the main report endpoint.
export const analyzeAPI = async (oldFile, newFile, dataset, pairId = null) => {
  let url = `${BACKEND_URL}/report?old=${encodeURIComponent(
    oldFile
  )}&new=${encodeURIComponent(newFile)}&dataset=${encodeURIComponent(dataset)}`;

  if (pairId) url += `&pair_id=${encodeURIComponent(pairId)}`;

  const res = await fetchWithRetry(url, {}, 1, 12_000);
  return unwrap(res);
};

// Get consumer services for a producer.
export const fetchConsumers = async (service, path = null) => {
  let url = `${BACKEND_URL}/consumers?service=${encodeURIComponent(service)}`;
  if (path) url += `&path=${encodeURIComponent(path)}`;
  const res = await fetchWithRetry(url, {}, 1);
  return unwrap(res);
};

// Fetch versioning metadata for a pair_id.
export const fetchVersioning = async (pairId) => {
  const url = `${BACKEND_URL}/versioning?pair_id=${encodeURIComponent(pairId)}`;
  const res = await fetchWithRetry(url, {}, 1);
  return unwrap(res);
};

// Get service graph from backend.
export const fetchGraph = async () => {
  const res = await fetchWithRetry(`${BACKEND_URL}/graph`, {}, 1);
  return unwrap(res);
};

// Submit training samples to backend.
export const trainModel = async (samples) => {
  const res = await fetchWithRetry(
    `${BACKEND_URL}/train`,
    {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      data: samples,
    },
    1,
    15_000
  );
  return unwrap(res);
};

// Batch-run reports for a dataset. Returns array of results.
// Picks probable v1/v2 pairs, falls back to consecutive pairs.
export async function fetchReportsBatch({ dataset, limit = 10, onProgress = null }) {
  const filesResp = await fetchFiles(dataset);
  const files = (filesResp && filesResp.samples) || [];

  const v1Candidates = files.filter((f) => /-v1\b/i.test(f) || /v1/i.test(f));
  const pairs = [];
  for (let f1 of v1Candidates) {
    const base = f1.replace(/-v1/i, "").replace(/v1/i, "");
    const f2 =
      files.find((x) => x !== f1 && x.includes(base) && /-v2\b/i.test(x)) ||
      files.find((x) => x !== f1 && x.includes(base) && /v2/i.test(x));
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
