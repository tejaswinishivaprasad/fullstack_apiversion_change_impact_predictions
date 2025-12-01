// src/App.jsx
import React, { useState, useEffect, useRef } from "react";
import {
  analyzeAPI,
  fetchDatasets,
  fetchFiles,
  fetchConsumers,
  fetchVersioning,
  fetchAce,
  exportAceFromServer,
} from "./services/api.js";

import {
  Container,
  Grid,
  Card,
  CardContent,
  Typography,
  TextField,
  Button,
  CircularProgress,
  MenuItem,
  Box,
  LinearProgress,
  Divider,
  IconButton,
  Tooltip,
  Switch,
  FormControlLabel,
  Stack,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  Menu,
} from "@mui/material";

import DownloadIcon from "@mui/icons-material/Download";
import ReplayIcon from "@mui/icons-material/Replay";
import PlayArrowIcon from "@mui/icons-material/PlayArrow";
import CsvFileIcon from "@mui/icons-material/Article";
import NdjsonIcon from "@mui/icons-material/Storage";

import {
  ResponsiveContainer,
  BarChart,
  Bar,
  XAxis,
  YAxis,
  Tooltip as ReTooltip,
  CartesianGrid,
  LineChart,
  Line,
  Legend,
} from "recharts";

// parse maybe JSON returned as string
const parseMaybeJson = (v, fallback) => {
  if (v == null) return fallback;
  if (typeof v === "string") {
    try {
      return JSON.parse(v);
    } catch {
      return fallback;
    }
  }
  return v;
};

// format service names for charts
const formatServiceName = (n = "") => {
  if (!n) return "";
  let s = String(n).replace(/^svc:/i, "").replace(/^ui:/i, "");
  s = s.replace(/[-_.]/g, " ").replace(/\s+/g, " ").trim();
  return s.length > 24 ? s.slice(0, 24) + "…" : s;
};

// tick renderer for charts
function CustomizedAxisTick({ x, y, payload, maxChars = 18 }) {
  const fullName = (payload && payload.payload && payload.payload.fullName) || payload?.value || "";
  const words = fullName.split(/\s+/);
  let line1 = "";
  let line2 = "";

  for (let w of words) {
    if ((line1 + " " + w).trim().length <= maxChars || !line1) {
      line1 = (line1 + " " + w).trim();
    } else {
      line2 = (line2 + " " + w).trim();
    }
  }

  if (!line2 && line1.length > maxChars) {
    line1 = line1.slice(0, maxChars - 1) + "…";
  } else if (line2 && line2.length > maxChars) {
    line2 = line2.slice(0, maxChars - 1) + "…";
  }

  const finalLine1 = line1 || fullName.slice(0, maxChars) + (fullName.length > maxChars ? "…" : "");
  const finalLine2 = line2 || "";

  return (
    <g transform={`translate(${x},${y})`}>
      <title>{fullName}</title>
      <text x={0} y={0} textAnchor="middle" style={{ fontSize: 12, fill: "#222" }}>
        <tspan x={0} dy="0" style={{ fontWeight: 500 }}>{finalLine1}</tspan>
        {finalLine2 ? <tspan x={0} dy="1.2em" style={{ fontWeight: 400 }}>{finalLine2}</tspan> : null}
      </text>
    </g>
  );
}

const toTop = (arr = []) =>
  (arr || [])
    .slice(0, 5)
    .map((x) => ({
      name: formatServiceName(x.service),
      fullName: x.service,
      risk: Number(x.risk_score ?? x.riskScore ?? x.risk ?? 0),
    }));

const HUMAN_TYPE = {
  ENDPOINT_ADDED: "Endpoint added",
  ENDPOINT_REMOVED: "Endpoint removed",
  PARAM_REQUIRED_ADDED: "Parameter made required",
  ENUM_NARROWED: "Enum narrowed",
  RESPONSE_CODE_REMOVED: "Response code removed",
  PARAM_ADDED: "Parameter added",
  PARAM_REMOVED: "Parameter removed",
  RESPONSE_SCHEMA_CHANGED: "Response schema changed",
  REQUESTBODY_SCHEMA_CHANGED: "Request body schema changed",
  UNKNOWN: "Change",
};

const explainChange = (d, rpt = {}) => {
  // normalize type to lowercase for robust matching
  const t = (d.type || "").toLowerCase();
  const det = d.detail;
  const path = d.path || (typeof det === "string" ? det : "");

  // tiny helper to make a readable short schema/type description
  const shortSchema = (s) => {
    if (!s) return "n/a";
    if (typeof s === "string") return s.slice(0, 80);
    if (typeof s === "object") {
      if (s.type) return String(s.type);
      if (s.$ref) return `ref:${String(s.$ref).split("/").pop()}`;
      if (s.properties && typeof s.properties === "object") return `${Object.keys(s.properties).length} fields`;
      return JSON.stringify(s).slice(0, 80);
    }
    return String(s).slice(0, 80);
  };

  // Prefer structured detail objects produced by the server
  if (det && typeof det === "object") {
    const kind = det.kind || "";

    if (kind === "param") {
      const name = det.name || "-";
      const loc = det.in || "-";
      if (det.required_changed) {
        if (det.required_changed.to) {
          return `Parameter '${name}' in ${loc} became required (was optional). Clients that omit it may receive validation errors (400).`;
        }
        return `Parameter '${name}' in ${loc} became optional (was required).`;
      }
      if (det.schema_changed) {
        const oldS = shortSchema(det.schema_changed.old);
        const newS = shortSchema(det.schema_changed.new);
        let extra = "";
        if (det.enum_narrowed && Array.isArray(det.enum_narrowed.removed) && det.enum_narrowed.removed.length) {
          extra = ` Enum narrowed: removed ${JSON.stringify(det.enum_narrowed.removed)}.`;
        } else if (det.enum_widened && Array.isArray(det.enum_widened.added) && det.enum_widened.added.length) {
          extra = ` Enum widened: added ${JSON.stringify(det.enum_widened.added)}.`;
        }
        return `Parameter '${name}' in ${loc} schema/type changed: ${oldS} → ${newS}.${extra} This may break clients that validate or parse this parameter.`;
      }
      return `Parameter '${name}' in ${loc} changed. Review ACE details for the exact diff.`;
    }

    if (kind === "response" || kind === "response_media" || kind === "response_schema") {
      const code = det.status_code || "-";
      const mt = det.media_type || det.media || "application/json";
      const oldS = shortSchema(det.old);
      const newS = shortSchema(det.new);
      return `Response ${code} (${mt}) schema changed: ${oldS} → ${newS}. Consumers parsing the payload may break if required fields or types were removed/changed.`;
    }

    if (kind === "requestbody" || kind === "requestbody_media" || kind === "requestbody_schema") {
      const mt = det.media_type || det.media || "application/json";
      const oldS = shortSchema(det.old);
      const newS = shortSchema(det.new);
      return `Request body (${mt}) schema changed: ${oldS} → ${newS}. Clients sending older payload shapes may fail server validation.`;
    }

    // generic structured fallback
    if (det.name || det.path) {
      return `Change: ${det.name || det.path}. Inspect ACE details for full context.`;
    }
  }

  // Fallback to older string-based details and normalized type cases
  switch (t) {
    case "endpoint_added":
      return `Adds a new API endpoint (${path}). New endpoints are normally non-breaking but increase surface area; check auth and shared DTOs.`;
    case "endpoint_removed":
      return `Removes an API endpoint (${path}). Removing endpoints is breaking for clients that used it; check consumers and replacement paths.`;
    case "param_required_added":
      return `A parameter became required (${path}). Clients omitting this will fail; this is a breaking change unless defaults exist.`;
    case "enum_narrowed":
      return `An enum or allowed-value set was narrowed (${path}). Clients using removed values may fail.`;
    case "response_code_removed":
      return `A response code was removed (${path}). Clients expecting that code may behave incorrectly; check client logic.`;
    case "param_added":
      return `A parameter was added (${path}). If optional it's non-breaking; if required it's breaking — check defaults.`;
    case "param_removed":
      return `A parameter was removed (${path}). Usually non-breaking but verify server validation behavior.`;
    case "response_schema_changed":
      return `Response schema changed (${path}). Consumers that parse the payload may break if fields or types changed.`;
    case "requestbody_schema_changed":
      return `Request body schema changed (${path}). Clients sending older shapes may fail; validate and communicate.`;
    default:
      // generic fallback: prefer the textual detail if available
      if (d.detail) {
        const txt = typeof d.detail === "string" ? d.detail : JSON.stringify(d.detail);
        return `Change detected (${d.type || "unknown"}): ${txt}`;
      }
      return `Change detected (${d.type || "unknown"}). Review consumers to judge impact.`;
  }
};


// Find best matching newer-version file for a selected 'oldFile' from a list of filenames.
// Returns filename string or null if none found.
export function findCorrespondingNewFile(oldFile, files = []) {
  if (!oldFile || !files || !files.length) return null;

  // normalize helper: extract base (remove version tokens) and version number if present
  const normalize = (fname) => {
    // keep extension and rest so we can compare correctly
    // look for version tokens like -v1, _v1, v1 (word boundary) optionally preceded by separators
    const m = fname.match(/(.*?)(?:[-_]?v(\d+)|\bv(\d+)\b)(.*)$/i);
    if (m) {
      const base = (m[1] + m[4]).replace(/(^[-_.]+|[-_.]+$)/g, "");
      const ver = parseInt(m[2] || m[3], 10);
      return { base, ver: Number.isFinite(ver) ? ver : null };
    }
    // no explicit version token — try to remove common suffix words like '.canonical' leaving extension
    const baseNoVer = fname.replace(/[-_.]?v\d+/i, "");
    return { base: baseNoVer, ver: null };
  };

  const oldNorm = normalize(oldFile);

  // Construct candidates that appear to share the same base (simple includes + normalized base compare)
  const candidates = files
    .filter((f) => f !== oldFile)
    .map((f) => ({ f, n: normalize(f) }))
    .filter(({ f, n }) => {
      // exact base match (ignoring version)
      const a = oldNorm.base.toLowerCase();
      const b = n.base.toLowerCase();
      return a === b || b.includes(a) || a.includes(b);
    });

  if (!candidates.length) return null;

  // If old has a version number, prefer the smallest candidate with version > old
  if (oldNorm.ver !== null) {
    const higher = candidates
      .filter(({ n }) => n.ver !== null && n.ver > oldNorm.ver)
      .sort((a, b) => a.n.ver - b.n.ver);
    if (higher.length) return higher[0].f;
  }

  // Otherwise try to find v2 specifically
  const v2 = candidates.find(({ f }) => /(?:[-_]?v2|\bv2\b)/i.test(f));
  if (v2) return v2.f;

  // fallback: pick candidate with highest version if versions exist
  const withVers = candidates.filter(({ n }) => n.ver !== null);
  if (withVers.length) {
    withVers.sort((a, b) => b.n.ver - a.n.ver);
    return withVers[0].f;
  }

  // last resort: try to pick file that shares most token overlap (base includes)
  return candidates[0].f || null;
}


function downloadJSON(obj, filename = "impact-report.json") {
  const blob = new Blob([JSON.stringify(obj ?? {}, null, 2)], { type: "application/json;charset=utf-8" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  a.click();
  URL.revokeObjectURL(url);
}

function downloadCSV(rows = [], filename = "impact-report.csv") {
  if (!rows || rows.length === 0) {
    const fallback = { message: "no rows to export" };
    downloadJSON(fallback, filename.replace(".csv", ".json"));
    return;
  }
  const headerCols = Object.keys(rows[0]);
  const lines = [];
  lines.push(headerCols.join(","));
  for (const row of rows) {
    const cells = headerCols.map((col) => {
      const raw = row[col] == null ? "" : String(row[col]);
      return JSON.stringify(raw);
    });
    lines.push(cells.join(","));
  }
  const csv = lines.join("\n");
  const blob = new Blob([csv], { type: "text/csv;charset=utf-8" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  a.click();
  URL.revokeObjectURL(url);
}

function computeSummaryFromReport(report) {
  const details = Array.isArray(report?.details) ? report.details : [];
  const aces = details.length;
  const backendImpactsCount = Array.isArray(report?.backend_impacts) ? report.backend_impacts.length : (report?.backend_impacts_count ?? 0);
  const frontendImpactsCount = Array.isArray(report?.frontend_impacts) ? report.frontend_impacts.length : (report?.frontend_impacts_count ?? 0);

  const flatRows = details.map((d, i) => ({
    id: i,
    change_type: d.type || "",
    path: d.path || (typeof d.detail === "string" ? d.detail : ""),
    ace_id: d.ace_id || "",
    predicted_risk: (d.risk_score ?? d.risk ?? "").toString(),
    consumers: (d.consumer_count ?? d.consumers ?? "").toString(),
  }));

  const calibration_bins = Array.isArray(report?.calibration_bins) && report.calibration_bins.length > 0
    ? report.calibration_bins
    : (() => {
        const scores = details.map((d) => Number(d.predicted_risk ?? d.risk_score ?? d.risk ?? report.predicted_risk ?? 0)).filter((x) => !Number.isNaN(x));
        if (!scores.length) return [];
        const buckets = 5;
        const binSize = 1 / buckets;
        const bins = new Array(buckets).fill(0).map((_, i) => ({ pred_mean: (i + 0.5) * binSize, empirical: 0, count: 0 }));
        scores.forEach((s) => {
          const idx = Math.min(Math.floor(s / binSize), buckets - 1);
          bins[idx].empirical += s;
          bins[idx].count += 1;
        });
        bins.forEach((b) => {
          if (b.count > 0) b.empirical = b.empirical / b.count;
          else b.empirical = b.pred_mean;
        });
        return bins;
      })();

  const confusion_matrix = Array.isArray(report?.confusion_matrix) ? report.confusion_matrix : null;
  const confusion_labels = Array.isArray(report?.confusion_labels) ? report.confusion_labels : (confusion_matrix ? ["Neg", "Pos"] : null);

  return {
    aces,
    backendImpactsCount,
    frontendImpactsCount,
    flatRows,
    calibration_bins,
    confusion_matrix,
    confusion_labels,
  };
}

// ACE modal component
function AceModal({ open, ace, loading, error, onClose }) {
  const [showDetailOnly, setShowDetailOnly] = React.useState(false);

  const extractAcePayload = () => {
    if (!ace) return null;
    if (ace.ace) return ace.ace;
    return ace;
  };

  const payload = extractAcePayload();

  // short one-line preview (what you saw before)
  const shortPreview = (() => {
    if (!payload) return "";
    if (payload.path) return String(payload.path);
    if (payload.detail && typeof payload.detail === "string") return payload.detail;
    // try to synthesize from common keys
    if (payload.type) return `${payload.type} ${payload.ace_id ? `(${payload.ace_id})` : ""}`;
    return "";
  })();

  const renderPrettyJSON = (obj) => {
    try {
      return JSON.stringify(obj, null, 2);
    } catch {
      return String(obj);
    }
  };

  const renderBody = () => {
    if (!payload) return "No ACE data available.";

    if (showDetailOnly) {
      // if detail exists show it (structured or string). If missing, fall back to full object.
      if (payload.detail !== undefined && payload.detail !== null) {
        if (typeof payload.detail === "object") return renderPrettyJSON(payload.detail);
        return String(payload.detail);
      }
      // fallback
      return renderPrettyJSON(payload);
    }

    // default: show full ACE object (so you always get everything)
    return renderPrettyJSON(payload);
  };

  // derive title text (keep same form you saw)
  const titleId = payload?.ace_id ? ` — ${payload.ace_id}` : "";
  const subtitle = shortPreview ? shortPreview : "";

  return (
    <Dialog open={!!open} onClose={onClose} maxWidth="md" fullWidth>
      <DialogTitle>
        ACE JSON{titleId}
      </DialogTitle>

      <DialogContent dividers style={{ minHeight: 140 }}>
        {loading ? (
          <Box display="flex" alignItems="center" justifyContent="center" padding={4}>
            <CircularProgress />
          </Box>
        ) : (
          <>
            {error && (
              <Box marginBottom={1}>
                <Typography variant="body2" color="error">{error}</Typography>
              </Box>
            )}

            {subtitle ? (
              <Box marginBottom={1}>
                <Typography variant="body2" style={{ fontFamily: "monospace", fontSize: 12, color: "#333" }}>
                  {subtitle}
                </Typography>
              </Box>
            ) : null}

            <Box display="flex" alignItems="center" gap={1} marginBottom={1}>
              <FormControlLabel
                control={<Switch checked={showDetailOnly} onChange={(e) => setShowDetailOnly(e.target.checked)} size="small" />}
                label={<Typography variant="caption">Show detail only</Typography>}
              />
              <Button
                size="small"
                onClick={() => {
                  // copy payload (full or detail) to clipboard
                  const toCopy = (() => {
                    if (showDetailOnly && payload?.detail !== undefined && payload?.detail !== null) {
                      return typeof payload.detail === "object" ? renderPrettyJSON(payload.detail) : String(payload.detail);
                    }
                    return renderPrettyJSON(payload);
                  })();
                  navigator.clipboard?.writeText(toCopy).then(() => {}, () => {});
                }}
              >
                Copy
              </Button>
            </Box>

            <pre style={{ whiteSpace: "pre-wrap", fontSize: 12, maxHeight: 480, overflow: "auto", margin: 0 }}>
              {renderBody()}
            </pre>
          </>
        )}
      </DialogContent>

      <DialogActions>
        <Button onClick={onClose} variant="outlined">Close</Button>
      </DialogActions>
    </Dialog>
  );
}


function renderBarLabel(props) {
  const { x, y, width, value } = props;
  const pct = Math.round((value ?? 0) * 100);
  return (
    <text x={x + width / 2} y={y - 6} textAnchor="middle" fontSize={12} fill="#222">
      {pct}%
    </text>
  );
}

export default function App() {
  const [datasets, setDatasets] = useState([]);
  const [dataset, setDataset] = useState("");
  const [availableFiles, setAvailableFiles] = useState([]);
  const [fileCount, setFileCount] = useState(0);
  const [oldSpec, setOld] = useState("");
  const [newSpec, setNew] = useState("");
  const [pairId, setPairId] = useState("");
  const [report, setReport] = useState(null);
  const [err, setErr] = useState("");
  const [loading, setLoading] = useState(false);
  const [autoSync, setAutoSync] = useState(true);
  const [consumers, setConsumers] = useState([]);
  const [lastRequest, setLastRequest] = useState(null);
  const latestReportRef = useRef(null);

  const [aceModalOpen, setAceModalOpen] = useState(false);
  const [aceModalItem, setAceModalItem] = useState(null);
  const [aceLoading, setAceLoading] = useState(false);
  const [aceError, setAceError] = useState(null);

  const [exportAnchor, setExportAnchor] = useState(null);
  const openExportMenu = (e) => setExportAnchor(e.currentTarget);
  const closeExportMenu = () => setExportAnchor(null);

  function closeAceModal() {
    setAceModalItem(null);
    setAceModalOpen(false);
    setAceLoading(false);
    setAceError(null);
  }

async function openAceModalFn(ace) {
  setAceError(null);
  setAceModalItem(null);

  if (!ace) {
    setAceError("No ACE data provided");
    return;
  }

  const aceId = ace.ace_id || ace.aceId || null;
  const pidFromReport =
      latestReportRef.current?.versioning?.pair_id ||
      latestReportRef.current?.backend?.pair_id ||
      latestReportRef.current?.metadata?.pair_id ||
      report?.versioning?.pair_id ||
      report?.backend?.pair_id ||
      report?.metadata?.pair_id ||
      null;

  let pid = pidFromReport || pairId || null;
  if (!pid && aceId && typeof aceId === "string" && aceId.includes("::")) {
    pid = aceId.split("::")[0];
  }

  // prevent duplicate fetch if already loading for same ace
  if (aceLoading && aceId) {
    console.warn("[UI] ACE already loading, skipping duplicate request", aceId);
    setAceModalOpen(true); // ensure modal visible
    return;
  }

  setAceModalOpen(true); // open modal immediately so UI doesn't block; we show spinner while loading
  if (aceId) {
    setAceLoading(true);
    try {
      const body = await fetchAce(pid, aceId);
      const payload = body && body.ace ? body.ace : body;
      if (!payload) {
        setAceModalItem(ace);
        setAceError("Server returned empty ACE payload; showing local diff instead.");
      } else if (payload.error || payload.message) {
        setAceModalItem(ace);
        setAceError(payload.message || payload.error || "Server returned an error while fetching ACE; showing local diff instead.");
      } else {
        setAceModalItem(payload || ace);
      }
    } catch (e) {
      console.warn("[UI] fetchAce failed:", e);
      let msg = "Failed to load original ACE from server; showing local diff item instead.";
      if (e && e.data && (e.data.message || e.data.error)) msg = e.data.message || e.data.error;
      else if (e && e.message) msg = e.message;
      setAceModalItem(ace);
      setAceError(msg);
    } finally {
      setAceLoading(false);
    }
  } else {
    setAceModalItem(ace);
  }
}


  useEffect(() => {
    (async () => {
      try {
        const dsRaw = await fetchDatasets();
        const dsList = (Array.isArray(dsRaw) ? dsRaw : parseMaybeJson(dsRaw, [])).map((key) => ({
          key,
          label: key.charAt(0).toUpperCase() + key.slice(1),
        }));
        setDatasets(dsList);
        if (dsList.length > 0) setDataset(dsList[0].key);
      } catch (e) {
        setErr("Failed to fetch datasets from backend.");
      }
    })();
  }, []);
  // keep newSpec auto-synced to oldSpec when enabled (covers manual old changes and late file loads)
  useEffect(() => {
    if (!autoSync) return;
    if (!oldSpec || !availableFiles || !availableFiles.length) return;
    const suggested = findCorrespondingNewFile(oldSpec, availableFiles);
    if (suggested && suggested !== newSpec) {
      setNew(suggested);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [oldSpec, autoSync, availableFiles]);


  useEffect(() => {
    if (!dataset) return;
    (async () => {
      try {
        const data = await fetchFiles(dataset);
        let parsed = parseMaybeJson(data, { samples: [], count: 0 });
        if (typeof parsed === "string") {
          try {
            parsed = JSON.parse(parsed);
          } catch {
            parsed = { samples: [], count: 0 };
          }
        }

        const samples = Array.isArray(parsed.samples) ? parsed.samples : [];
        setAvailableFiles(samples);
        setFileCount(Number(parsed.count || samples.length || 0));

        if (samples.length >= 2) {
          const firstV1 = samples.find((s) => /(?:[-_.]?v1|\bv1\b)/i.test(s)) || samples[0];
          const suggestedV2 = findCorrespondingNewFile(firstV1, samples);
          const v1 = firstV1;
          const v2 = suggestedV2 || samples.find((s) => s !== v1) || samples[0];
          setOld(v1);
          setNew(v2);
        }
        else if (samples.length === 1) {
          setOld(samples[0]);
          setNew(samples[0]);
        } else {
          setOld("");
          setNew("");
        }

        setAutoSync(true);
      } catch (e) {
        console.error("[UI] loadFiles error:", e);
        setErr(`Failed to fetch files for ${dataset}`);
        setAvailableFiles([]);
        setFileCount(0);
        setOld("");
        setNew("");
      }
    })();
  }, [dataset]);

  const run = async (opts = {}) => {
  setErr("");
  setReport(null);
  setLoading(true);
  const usePairId = opts.usePair ?? !!pairId;

  const isBadPid = (p) =>
    !p ||
    String(p).trim() === "" ||
    ["n/a", "na", "null", "undefined"].includes(String(p).toLowerCase());

  try {
    const data = await analyzeAPI(oldSpec, newSpec, dataset, usePairId ? pairId : null);
    const parsed = parseMaybeJson(data, null) || {};
    parsed.metadata = parsed.metadata || {};
    if (!parsed.metadata.dataset) parsed.metadata.dataset = dataset;
    if (!parsed.metadata.file_name) parsed.metadata.file_name = newSpec || "";
    if (!parsed.metadata.generated_at) parsed.metadata.generated_at = new Date().toISOString();
    parsed.metadata.commit_hash = parsed.metadata.commit_hash || parsed.versioning?.commit || parsed.versioning?.git_commit || null;
    parsed.metadata.repo_url =
      parsed.metadata.repo_url ||
      (parsed.metadata.repo_owner && parsed.metadata.repo_name
        ? `https://github.com/${parsed.metadata.repo_owner}/${parsed.metadata.repo_name}`
        : parsed.metadata.repo_url || null);

    setReport(parsed);
    latestReportRef.current = parsed;

    // store lastRequest in snake_case pair_id so API wrapper expects that param
    setLastRequest({ oldSpec, newSpec, dataset, pair_id: usePairId ? pairId : null });

    // ---------- robust pair_id resolution ----------
    // Prefer authoritative locations: versioning, backend, metadata, then fallback to logs
    let resolvedPid =
      parsed.versioning?.pair_id ||
      parsed.backend?.pair_id ||
      parsed.metadata?.pair_id ||
      parsed.metadata?.pairId ||
      null;

    if (!resolvedPid || isBadPid(resolvedPid)) {
      // fallback to parsing logs only if nothing authoritative found
      if (Array.isArray(parsed.logs) && parsed.logs.length > 0) {
        const logEntry = parsed.logs.find((l) => typeof l === "string" && l.includes("pair_id="));
        if (logEntry) {
          const m = /pair_id=([^\s,;]+)/.exec(logEntry);
          if (m && m[1]) {
            const pidFromLog = m[1];
            if (!isBadPid(pidFromLog)) resolvedPid = pidFromLog;
            else console.info("[UI] ignored invalid pid from logs:", pidFromLog);
          }
        }
      }
    }

    if (resolvedPid && !isBadPid(resolvedPid)) {
      // update UI state with authoritative pid
      setPairId(resolvedPid);

      // if server didn't return versioning payload, try to fetch it (safe, idempotent)
      const hasVersioning = parsed.versioning && Object.keys(parsed.versioning || {}).length > 0;
      if (!hasVersioning) {
        try {
          const v = await fetchVersioning(resolvedPid);
          // merge versioning safely into current report
          setReport((r) => ({ ...(r || {}), versioning: v || {} }));
          latestReportRef.current = { ...(latestReportRef.current || {}), versioning: v || {} };
        } catch (verr) {
          // don't fail the whole run if version lookup fails; just log
          console.warn("[UI] fetchVersioning failed for pid=", resolvedPid, verr);
        }
      }
    } else {
      // leave pairId alone (or cleared) if nothing valid found
      if (!resolvedPid) {
        // don't overwrite a manually entered pairId if usePairId was true
        if (!usePairId) setPairId("");
      }
    }

    // ---------- fetch consumers if backend impacts exist ----------
    if (parsed && parsed.backend_impacts && parsed.backend_impacts.length > 0) {
      try {
        const svc = parsed.backend_impacts[0].service;
        const svcClean = (svc || "").replace(/^svc:/i, "").replace(/^ui:/i, "");
        const c = await fetchConsumers(svcClean);
        setConsumers(parseMaybeJson(c, []));
      } catch (cerr) {
        console.warn("[UI] fetchConsumers failed", cerr);
        setConsumers([]);
      }
    } else {
      setConsumers([]);
    }
  } catch (e) {
    // show structured server messages if present
    let msg = "Failed to analyze. Check backend logs.";
    if (e && e.data) {
      if (typeof e.data === "string") msg = e.data;
      else if (e.data.message) msg = e.data.message;
      else if (e.data.detail) msg = Array.isArray(e.data.detail) ? e.data.detail.join("\n") : e.data.detail;
      else msg = JSON.stringify(e.data);
    } else if (e && e.message) {
      msg = e.message;
    }
    console.error("[UI] run() failed:", e);
    setErr(msg);
  } finally {
    setLoading(false);
  }
};


  const retry = async () => {
    if (!lastRequest) return;
    setErr("");
    setLoading(true);
    try {
      const r = await analyzeAPI(lastRequest.oldSpec, lastRequest.newSpec, lastRequest.dataset, lastRequest.pair_id);
      const parsed = parseMaybeJson(r, null) || {};
      parsed.metadata = parsed.metadata || {};
      if (!parsed.metadata.generated_at) parsed.metadata.generated_at = new Date().toISOString();
      setReport(parsed);
      latestReportRef.current = parsed;
    } catch (e) {
      let msg = "Retry failed";
      if (e && e.data && (e.data.message || e.data.error)) msg = e.data.message || e.data.error;
      else if (e && e.message) msg = e.message;
      setErr(msg);
    } finally {
      setLoading(false);
    }
  };

  const exportReport = () => {
    const r = latestReportRef.current || report;
    if (!r) {
      alert("No report to export");
      return;
    }
    const copy = { ...r };
    copy.metadata = copy.metadata || {};
    copy.metadata.exported_at = new Date().toISOString();
    if (!copy.metadata.dataset) copy.metadata.dataset = dataset;
    const name = `report-${(copy.metadata.dataset || dataset)}-${(copy.metadata.commit_hash || copy.backend?.producer || "unknown")}.json`;
    downloadJSON(copy, name);
  };

  const exportCSVFromReportDetails = () => {
    const r = latestReportRef.current || report;
    if (!r) {
      alert("No report to export");
      return;
    }
    const summary = computeSummaryFromReport(r);
    const name = `report-${(r.metadata?.dataset || dataset)}-${Date.now()}.csv`;
    downloadCSV(summary.flatRows, name);
  };

  const exportNDJSONBatch = () => {
    const r = latestReportRef.current || report;
    if (!r) {
      alert("No report to export");
      return;
    }
    const nd = JSON.stringify(r) + "\n";
    const blob = new Blob([nd], { type: "application/x-ndjson" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `report-${(r.metadata?.dataset || dataset)}-${Date.now()}.ndjson`;
    a.click();
    URL.revokeObjectURL(url);
  };

  const computed = computeSummaryFromReport(report || {});
  const acesCount = computed.aces;
  const backendCount = computed.backendImpactsCount;
  const frontendCount = computed.frontendImpactsCount;
  const calibrationBins = computed.calibration_bins;
  const confusionMatrix = computed.confusion_matrix;
  const confusionLabels = computed.confusion_labels;

  const topB = toTop(report?.backend_impacts || []);
  const topF = toTop(report?.frontend_impacts || []);

  const riskLevel = (score) => {
    const s = score ?? 0;
    if (s >= 0.7) return { label: "High", color: "error" };
    if (s >= 0.4) return { label: "Medium", color: "warning" };
    return { label: "Low", color: "success" };
  };

  const provenanceLine = (r) => {
    if (!r) return "";
    const pid =
      r.versioning?.pair_id ||
      r.backend?.pair_id ||
      r.metadata?.pair_id ||
      r.logs?.find((l) => typeof l === "string" && l.includes("pair_id="))?.match(/pair_id=([^\s,;]+)/)?.[1] ||
      null;
    const risk = (r.predicted_risk ?? r.risk_score ?? 0).toFixed(2);
    const gen = r.metadata?.generated_at || "";
    return [pid ? `pair=${pid}` : null, `risk=${risk}`, gen ? `Generated ${gen}` : null].filter(Boolean).join(" • ");
  };


  function compactAceSummary(detail) {
  if (!detail) return "";
  if (typeof detail === "string") return detail.slice(0, 120);

  if (typeof detail === "object") {
    const kind = detail.kind;
    if (kind === "param") {
      return `Param '${detail.name}' in ${detail.in}`;
    }
    if (kind && kind.includes("response")) {
      return `Response change (${detail.media_type || "json"})`;
    }
    if (kind && kind.includes("requestbody")) {
      return `Request body change (${detail.media_type || "json"})`;
    }
    // fallback for odd cases
    return JSON.stringify(detail).slice(0, 120);
  }

  return String(detail).slice(0, 120);
}


  const renderHumanExplanation = (rpt) => {
    if (!rpt) return null;
    const score = rpt.predicted_risk ?? rpt.risk_score ?? 0;
    const band = riskLevel(score).label;
    const numChanges = (rpt.details && rpt.details.length) || 0;
    const backendCountLocal = (rpt.backend_impacts && rpt.backend_impacts.length) || 0;
    const frontendCountLocal = (rpt.frontend_impacts && rpt.frontend_impacts.length) || 0;
    const confOverall = Math.round((rpt.confidence?.overall ?? 0) * 100);
    const confBackend = Math.round((rpt.confidence?.backend ?? 0) * 100);
    const confFrontend = Math.round((rpt.confidence?.frontend ?? 0) * 100);

    const bullets = [
      `Risk: ${band} (${(score).toFixed(2)}) — confidence overall ${confOverall}%`,
      `${numChanges} API change${numChanges !== 1 ? "s" : ""} detected`,
      `${backendCountLocal} backend module${backendCountLocal !== 1 ? "s" : ""} potentially impacted`,
      `${frontendCountLocal} frontend module${frontendCountLocal !== 1 ? "s" : ""} potentially impacted`,
    ];

    const versionLines = [];
    if (rpt.versioning && Object.keys(rpt.versioning).length > 0) {
      if (rpt.versioning.breaking_vs_semver) versionLines.push("Versioning: breaking vs semver detected");
      else versionLines.push(`Versioning: ${rpt.versioning.semver_old || "n/a"} → ${rpt.versioning.semver_new || "n/a"} (${rpt.versioning.semver_delta || "n/a"})`);
    }

    return (
      <>
        <Typography variant="body1" style={{ marginBottom: 8 }}>
          <strong>Summary:</strong> Predicted risk is <strong>{band}</strong> ({score.toFixed(2)}). Confidence: overall {confOverall}%, backend {confBackend}%, frontend {confFrontend}%.
        </Typography>

        <ul style={{ marginTop: 8 }}>
          {bullets.map((b, i) => (
            <li key={i}><Typography variant="body2">{b}</Typography></li>
          ))}
          {versionLines.map((v, i) => (
            <li key={`v-${i}`}><Typography variant="body2">{v}</Typography></li>
          ))}
        </ul>

        <Divider style={{ margin: "8px 0" }} />

        <Typography variant="subtitle2" style={{ marginBottom: 6 }}>Backend impacts </Typography>
        {rpt.backend_impacts && rpt.backend_impacts.length > 0 ? (
          <ul>
            {rpt.backend_impacts.slice(0, 6).map((b, idx) => (
              <li key={`b-${idx}`}><Typography variant="body2">{b.service} — risk {(b.risk_score ?? b.risk ?? 0).toFixed(2)}</Typography></li>
            ))}
          </ul>
        ) : (
          <Typography variant="body2">No backend impacts detected.</Typography>
        )}

        <Typography variant="subtitle2" style={{ marginTop: 8, marginBottom: 6 }}>Frontend impacts </Typography>
        {rpt.frontend_impacts && rpt.frontend_impacts.length > 0 ? (
          <ul>
            {rpt.frontend_impacts.slice(0, 6).map((f, idx) => (
              <li key={`f-${idx}`}><Typography variant="body2">{f.service} — risk {(f.risk_score ?? f.risk ?? 0).toFixed(2)}</Typography></li>
            ))}
          </ul>
        ) : (
          <Typography variant="body2">No frontend impacts detected.</Typography>
        )}

        <Divider style={{ margin: "8px 0" }} />

        <Typography variant="subtitle2">Change items</Typography>
        <ol>
          {(rpt.details || []).map((d, i) => {
            const human = HUMAN_TYPE[d.type] || d.type || "Change";
            const aceId = d.ace_id ? ` [${d.ace_id}]` : "";
            const reason = explainChange(d, rpt);
            const shortDetail = compactAceSummary(d);
            return (
              <li key={i} style={{ marginBottom: 8 }}>
                <Typography variant="body2">
                  <strong
                    style={{ cursor: "pointer", color: "#1565c0" }}
                    onClick={() => openAceModalFn(d)}
                  >
                    {human}
                  </strong>{aceId} — {shortDetail}
                </Typography>

                <Typography variant="body2" style={{ color: "#555", marginTop: 4 }}>
                  {reason}
                </Typography>

                <div style={{ marginTop: 6 }}>
                  <Button size="small" variant="text" onClick={() => openAceModalFn(d)}>View ACE</Button>
                </div>
              </li>
            );
          })}
        </ol>
      </>
    );
  };

  return (
    <Container maxWidth="lg" style={{ padding: "1rem 0 2rem 0" }}>
      <Typography variant="h5" align="center" gutterBottom>
        Impact AI — Prototype Dashboard {dataset ? `| ${dataset.toUpperCase()}` : ""}
      </Typography>

      <Card style={{ marginBottom: 12 }}>
        <CardContent>
          <Grid container spacing={2} alignItems="center">
            <Grid item xs={12} md={3}>
              <TextField
                select
                fullWidth
                size="small"
                label="Dataset"
                value={dataset}
                onChange={(e) => setDataset(e.target.value)}
                InputLabelProps={{ shrink: true }}
                inputProps={{ style: { textOverflow: "ellipsis", whiteSpace: "nowrap", overflow: "hidden" } }}
              >
                {(datasets || []).map((ds) => (
                  <MenuItem key={ds.key} value={ds.key}>
                    {ds.label}
                  </MenuItem>
                ))}
              </TextField>
              <Typography variant="caption">{fileCount > 0 ? `files: ${fileCount}` : "no files"}</Typography>
            </Grid>

            <Grid item xs={12} md={3}>
              <TextField
                select
                fullWidth
                size="small"
                label="Old Spec"
                value={oldSpec}
                onChange={(e) => {
                  const chosen = e.target.value;
                  setOld(chosen);
                  if (autoSync && availableFiles && availableFiles.length) {
                    const suggested = findCorrespondingNewFile(chosen, availableFiles);
                    if (suggested) setNew(suggested);
                  }
                }}

                SelectProps={{ MenuProps: { PaperProps: { style: { maxHeight: 360 } } } }}
                InputLabelProps={{ shrink: true }}
                inputProps={{ style: { textOverflow: "ellipsis", whiteSpace: "nowrap", overflow: "hidden" } }}
              >
                {(availableFiles || []).map((f, i) => (
                  <MenuItem key={i} value={f}>
                    {f}
                  </MenuItem>
                ))}
              </TextField>
            </Grid>

            <Grid item xs={12} md={3}>
              <TextField
                select
                fullWidth
                size="small"
                label="New Spec"
                value={newSpec}
                onChange={(e) => {
                  setNew(e.target.value);
                  setAutoSync(false);
                }}
                SelectProps={{ MenuProps: { PaperProps: { style: { maxHeight: 360 } } } }}
                InputLabelProps={{ shrink: true }}
                inputProps={{ style: { textOverflow: "ellipsis", whiteSpace: "nowrap", overflow: "hidden" } }}
              >
                {(availableFiles || []).map((f, i) => (
                  <MenuItem key={i} value={f}>
                    {f}
                  </MenuItem>
                ))}
              </TextField>
            </Grid>

            <Grid item xs={12} md={2}>
              <TextField size="small" fullWidth label="Pair ID (optional)" value={pairId} onChange={(e) => setPairId(e.target.value)} InputLabelProps={{ shrink: true }} />
            </Grid>

            <Grid item xs={12} md={1} style={{ display: "flex", alignItems: "center", justifyContent: "flex-end" }}>
              <Tooltip title="Run analysis">
                <span>
                  <Button
                    variant="contained"
                    color="success"
                    onClick={() => run({ usePair: !!pairId })}
                    disabled={loading || !oldSpec || !newSpec}
                    style={{ minWidth: 44, padding: 8 }}
                  >
                    {loading ? <CircularProgress size={18} /> : <PlayArrowIcon />}
                  </Button>
                </span>
              </Tooltip>
            </Grid>

          </Grid>
        </CardContent>
      </Card>

      {err && (
        <Card style={{ background: "#fff3f0", marginBottom: 12 }}>
          <CardContent>
            <Typography color="error">{String(err)}</Typography>
          </CardContent>
        </Card>
      )}

      {report && (
        <>
          <Box marginBottom={2}>
            <Card style={{ background: "#f6f7f9" }}>
              <CardContent>
                <Grid container spacing={2} alignItems="center">
                  <Grid item xs={12} style={{ display: "flex", justifyContent: "space-between", gap: 12, flexWrap: "wrap" }}>
                    <Typography
                      variant="body2"
                      style={{
                        whiteSpace: "nowrap",
                        overflow: "hidden",
                        textOverflow: "ellipsis",
                        maxWidth: "70%",
                      }}
                      title={`${provenanceLine(report)} • ACES ${report.summary_counts?.aces ?? acesCount} | BE ${report.summary_counts?.backend_impacts ?? backendCount} | FE ${report.summary_counts?.frontend_impacts ?? frontendCount}`}
                    >
                      {provenanceLine(report)}
                    </Typography>

                    <Box style={{ display: "flex", gap: 8, alignItems: "center" }}>
                      <Tooltip title="Export menu">
                        <IconButton onClick={openExportMenu} size="small">
                          <DownloadIcon />
                        </IconButton>
                      </Tooltip>

                      <Menu anchorEl={exportAnchor} open={Boolean(exportAnchor)} onClose={closeExportMenu}>
                        <MenuItem onClick={() => { closeExportMenu(); exportReport(); }}>
                          <DownloadIcon style={{ marginRight: 8 }} /> Export JSON (report)
                        </MenuItem>
                        <MenuItem onClick={() => { closeExportMenu(); exportCSVFromReportDetails(); }}>
                          <CsvFileIcon style={{ marginRight: 8 }} /> Export CSV (details)
                        </MenuItem>
                        <MenuItem onClick={() => { closeExportMenu(); exportNDJSONBatch(); }}>
                          <NdjsonIcon style={{ marginRight: 8 }} /> Export NDJSON (report)
                        </MenuItem>
                      </Menu>

                      <Tooltip title="Retry last analysis">
                        <IconButton onClick={retry} size="small">
                          <ReplayIcon />
                        </IconButton>
                      </Tooltip>
                    </Box>
                  </Grid>

                  <Grid item xs={12}>
                    <Grid container spacing={2} alignItems="center">
                      <Grid item xs={12} md={3}>
                        <Typography variant="subtitle2">ACES</Typography>
                        <Typography variant="h6">{report.summary_counts?.aces ?? acesCount}</Typography>
                      </Grid>

                      <Grid item xs={12} md={3}>
                        <Typography variant="subtitle2">Backend impacts</Typography>
                        <Typography variant="h6">{report.summary_counts?.backend_impacts ?? backendCount}</Typography>
                      </Grid>

                      <Grid item xs={12} md={3}>
                        <Typography variant="subtitle2">Frontend impacts</Typography>
                        <Typography variant="h6">{report.summary_counts?.frontend_impacts ?? frontendCount}</Typography>
                      </Grid>
                    </Grid>
                  </Grid>
                </Grid>
              </CardContent>
            </Card>
          </Box>

          <Grid container spacing={2} marginBottom={2} alignItems="center">
            <Grid item xs={12} md={6}>
              <Card>
                <CardContent>
                  <Typography variant="subtitle1">Predicted Risk</Typography>
                  <Box display="flex" alignItems="center" marginTop={1}>
                    <Box position="relative" display="inline-flex" marginRight={2}>
                      <CircularProgress
                        variant="determinate"
                        value={(report.predicted_risk ?? 0) * 100}
                        size={78}
                        thickness={5}
                        style={{
                          color:
                            (report.predicted_risk ?? 0) >= 0.7
                              ? "#e53935"
                              : (report.predicted_risk ?? 0) >= 0.4
                              ? "#ffb300"
                              : "#43a047",
                        }}
                      />
                      <Box position="absolute" top={0} left={0} right={0} bottom={0} display="flex" alignItems="center" justifyContent="center">
                        <Typography variant="h6">{(report.predicted_risk ?? 0).toFixed(2)}</Typography>
                      </Box>
                    </Box>

                    <Box flex="1">
                      <Typography variant="body2" style={{ marginBottom: 6 }}>{riskLevel(report.predicted_risk).label}</Typography>
                      <Typography variant="body2" color="textSecondary">
                        {((report.details || []).length || 0)} change{((report.details || []).length !== 1) ? "s" : ""} detected
                      </Typography>
                    </Box>
                  </Box>
                </CardContent>
              </Card>
            </Grid>

            <Grid item xs={12} md={6}>
              <Card>
                <CardContent>
                  <Typography variant="subtitle1">Confidence</Typography>
                  <Box marginTop={1}>
                    <Typography variant="caption">Overall confidence</Typography>
                    <LinearProgress variant="determinate" value={(report.confidence?.overall ?? 0) * 100} style={{ height: 10, borderRadius: 6, marginBottom: 8 }} />

                    <Stack direction="row" spacing={2} alignItems="center">
                      <Box width="50%">
                        <Typography variant="caption">Backend</Typography>
                        <LinearProgress variant="determinate" value={(report.confidence?.backend ?? 0) * 100} style={{ height: 8, borderRadius: 6 }} />
                      </Box>
                      <Box width="50%">
                        <Typography variant="caption">Frontend</Typography>
                        <LinearProgress variant="determinate" value={(report.confidence?.frontend ?? 0) * 100} style={{ height: 8, borderRadius: 6 }} />
                      </Box>
                    </Stack>
                  </Box>
                </CardContent>
              </Card>
            </Grid>
          </Grid>

          <Grid container spacing={2} marginBottom={2}>
            <Grid item xs={12} md={4}>
              <Card>
                <CardContent style={{ padding: 4 }}>
                  <Typography variant="h6">Backend Impacts</Typography>
                  {topB.length === 0 ? (
                    <Typography variant="body2" color="textSecondary">No backend impacts detected.</Typography>
                  ) : (
                    <ResponsiveContainer width="100%" height={260}>
                      <BarChart data={topB} margin={{ top: 4, right: 8, left: 0, bottom: 40 }}>
                        <CartesianGrid strokeDasharray="3 3" />
                        <XAxis dataKey="name" tick={<CustomizedAxisTick maxChars={18} />} interval={0} height={48} tickMargin={10} />
                        <YAxis domain={[0, 1]} tickFormatter={(v) => `${Math.round(v * 100)}%`} />
                        <ReTooltip formatter={(value) => `${Math.round(value * 100)}%`} labelFormatter={(label, payload) => (payload && payload[0] && payload[0].payload.fullName) || label} />
                        <Bar dataKey="risk" barSize={26} label={renderBarLabel} isAnimationActive={false} />
                      </BarChart>
                    </ResponsiveContainer>
                  )}
                </CardContent>
              </Card>
            </Grid>

            <Grid item xs={12} md={4}>
              <Card>
                <CardContent style={{ padding: 4 }}>
                  <Typography variant="h6">Frontend Impacts</Typography>
                  {topF.length === 0 ? (
                    <Typography variant="body2" color="textSecondary">No frontend impacts detected.</Typography>
                  ) : (
                    <ResponsiveContainer width="100%" height={260}>
                      <BarChart data={topF} margin={{ top: 4, right: 8, left: 0, bottom: 40 }}>
                        <CartesianGrid strokeDasharray="3 3" />
                        <XAxis dataKey="name" tick={<CustomizedAxisTick maxChars={18} />} interval={0} height={48} tickMargin={10} />
                        <YAxis domain={[0, 1]} tickFormatter={(v) => `${Math.round(v * 100)}%`} />
                        <ReTooltip formatter={(value) => `${Math.round(value * 100)}%`} labelFormatter={(label, payload) => (payload && payload[0] && payload[0].payload.fullName) || label} />
                        <Bar dataKey="risk" barSize={26} label={renderBarLabel} isAnimationActive={false} />
                      </BarChart>
                    </ResponsiveContainer>
                  )}
                </CardContent>
              </Card>
            </Grid>

            <Grid item xs={12} md={4}>
              <Card>
                <CardContent style={{ padding: 4 }}>
                  <Typography variant="h6">Calibration</Typography>
                  <ResponsiveContainer width="100%" height={260}>
                    <LineChart
                      data={calibrationBins.map((b, i) => ({
                        bin: `${Math.round((b.pred_mean ?? 0) * 100)}%`,
                        empirical: (b.empirical ?? 0) * 100,
                        ideal: ((i / Math.max(1, calibrationBins.length - 1)) * 100)
                      }))}
                      margin={{ top: 4, right: 12, left: 8, bottom: 40 }}
                    >
                      <CartesianGrid strokeDasharray="3 3" />
                      <XAxis dataKey="bin" height={40} tickMargin={10} />
                      <YAxis domain={[0, 100]} tickFormatter={(v) => `${v}%`} />
                      <ReTooltip formatter={(v) => `${v}%`} />
                      <Legend verticalAlign="bottom" height={20} />
                      <Line type="monotone" dataKey="empirical" stroke="#1976d2" strokeWidth={2} dot={{ r: 3 }} />
                      <Line type="linear" dataKey="ideal" stroke="#999" strokeDasharray="4 4" dot={false} />
                    </LineChart>
                  </ResponsiveContainer>
                </CardContent>
              </Card>
            </Grid>
          </Grid>

          <Card style={{ marginBottom: 12 }}>
            <CardContent>
              <Typography variant="h6">AI Explanation</Typography>
              <Box marginTop={1}>{renderHumanExplanation(report)}</Box>
            </CardContent>
          </Card>

          {report.versioning && (
            <Card style={{ marginBottom: 12 }}>
              <CardContent>
                <Typography variant="h6">Versioning / Metadata</Typography>
                <pre style={{ whiteSpace: "pre-wrap", margin: 0 }}>{JSON.stringify(report.versioning, null, 2)}</pre>
              </CardContent>
            </Card>
          )}

          {consumers && consumers.length > 0 && (
            <Card style={{ marginBottom: 12 }}>
              <CardContent>
                <Typography variant="h6">Consumers (sample)</Typography>
                <pre style={{ whiteSpace: "pre-wrap", margin: 0 }}>{JSON.stringify(consumers, null, 2)}</pre>
              </CardContent>
            </Card>
          )}
        </>
      )}

      <AceModal
        open={aceModalOpen}
        ace={aceModalItem}
        loading={aceLoading}
        error={aceError}
        onClose={closeAceModal}
      />
    </Container>
  );
}
