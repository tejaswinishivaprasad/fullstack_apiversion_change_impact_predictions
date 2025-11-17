// src/App.jsx
import React, { useState, useEffect, useRef } from "react";
import {
  analyzeAPI,
  fetchDatasets,
  fetchFiles,
  fetchConsumers,
  trainModel,
  fetchVersioning,
  fetchReportsBatch,
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
  Table,
  TableBody,
  TableCell,
  TableRow,
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
import CodeIcon from "@mui/icons-material/Code";
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

// basic helpers
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

const formatServiceName = (n = "") => {
  if (!n) return "";
  let s = String(n).replace(/^svc:/i, "").replace(/^ui:/i, "");
  s = s.replace(/[-_.]/g, " ").replace(/\s+/g, " ").trim();
  return s.length > 24 ? s.slice(0, 24) + "…" : s;
};

// small tick renderer for charts
const CustomizedAxisTick = ({ x, y, payload }) => {
  const short = String(payload?.value ?? "");
  const fullName = (payload && payload.payload && payload.payload.fullName) || short;
  return (
    <g transform={`translate(${x},${y})`}>
      <title>{fullName}</title>
      <text
        x={0}
        y={0}
        dy={16}
        textAnchor="end"
        transform="rotate(-35)"
        style={{ fontSize: 12, fill: "#333" }}
      >
        {short}
      </text>
    </g>
  );
};

const toTop = (arr = []) =>
  (arr || [])
    .slice(0, 5)
    .map((x) => ({
      name: formatServiceName(x.service),
      fullName: x.service,
      risk: Number(x.risk_score ?? x.riskScore ?? x.risk ?? 0),
    }));

// human readable mapping for ace types
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

// short human explanations for change types
const explainChange = (d, rpt = {}) => {
  const t = (d.type || "").toUpperCase();
  const path = d.path || d.detail || "";
  switch (t) {
    case "ENDPOINT_ADDED":
    case "endpoint_added":
      return `This adds a new API endpoint (${path}). New endpoints are normally non-breaking, but they increase the public surface area. New consumers may start depending on this endpoint, which raises maintenance and compatibility risk for future changes; verify if new endpoints require auth or shared DTOs that other services rely on.`;
    case "ENDPOINT_REMOVED":
    case "endpoint_removed":
      return `This removes an endpoint (${path}). Removing endpoints is breaking for any client that used it. Check versioning and consumer usage; migrating clients may need updates, and CI gate should block such removals unless a replacement exists.`;
    case "PARAM_REQUIRED_ADDED":
    case "param_required_added":
      return `A parameter became required (${path}). Clients that don't send this parameter will fail. This is a classic breaking change; confirm semantic versioning and communicate to consumers.`;
    case "ENUM_NARROWED":
      return `An enum or allowed-value set was narrowed (${path}). Clients that previously used removed values may fail; this can be breaking if clients send the excluded values.`;
    case "RESPONSE_CODE_REMOVED":
    case "response_code_removed":
      return `A response code was removed (${path}). Clients handling that specific response may behave incorrectly; check client-side error handling and fallback logic.`;
    case "PARAM_ADDED":
      return `A parameter was added (${path}). If the parameter is optional this is non-breaking; if clients must supply it, it becomes breaking. Check whether it’s required and whether server-side defaults exist.`;
    case "PARAM_REMOVED":
      return `A parameter was removed (${path}). Clients still sending the parameter may be ignored but this is usually non-breaking; verify contract expectations and server validation behavior.`;
    case "RESPONSE_SCHEMA_CHANGED":
    case "response_schema_changed":
      return `Response schema changed (${path}). Consumers parsing responses may break if fields were renamed/removed or types changed; check backward-compatibility and add migration notes.`;
    case "REQUESTBODY_SCHEMA_CHANGED":
    case "requestbody_schema_changed":
      return `Request body schema changed (${path}). Clients sending older request shapes may fail or be rejected; treat as potentially breaking and validate with consumers.`;
    default:
      return d.detail
        ? `Change detected (${d.type}): ${d.detail}. Review the consumer contracts for this path to determine risk.`
        : `Change detected (${d.type}). Review consumers of this API to understand the practical impact.`;
  }
};

// ----------------- download helpers -----------------
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

// ----------------- client side aggregators -----------------
function computeSummaryFromReport(report) {
  const details = Array.isArray(report?.details) ? report.details : [];
  const aces = details.length;
  const backendImpactsCount = Array.isArray(report?.backend_impacts) ? report.backend_impacts.length : (report?.backend_impacts_count ?? 0);
  const frontendImpactsCount = Array.isArray(report?.frontend_impacts) ? report.frontend_impacts.length : (report?.frontend_impacts_count ?? 0);

  const flatRows = details.map((d, i) => ({
    id: i,
    change_type: d.type || "",
    path: d.path || d.detail || "",
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

// ----------------- small UI parts -----------------
function CalibrationChart({ bins = [] }) {
  if (!bins || !bins.length) {
    return (
      <Typography variant="body2" color="textSecondary">
        No calibration data available.
      </Typography>
    );
  }

  const labels = bins.map((b) => `${Math.round((b.pred_mean ?? 0) * 100)}%`);
  const data = bins.map((b, i) => ({
    bin: labels[i],
    empirical: (b.empirical ?? 0) * 100,
    ideal: ((i / Math.max(1, bins.length - 1)) * 100),
  }));

  return (
    <ResponsiveContainer width="100%" height={240}>
      <LineChart data={data} margin={{ top: 6, right: 12, bottom: 30, left: 8 }}>
        <CartesianGrid strokeDasharray="3 3" />
        <XAxis dataKey="bin" angle={-35} textAnchor="end" height={60} />
        <YAxis domain={[0, 100]} tickFormatter={(v) => `${v}%`} />
        <ReTooltip formatter={(v) => `${v}%`} />
        <Legend verticalAlign="bottom" height={36} />
        <Line type="monotone" dataKey="empirical" stroke="#1976d2" strokeWidth={2} dot={{ r: 3 }} />
        <Line type="linear" dataKey="ideal" stroke="#999" strokeDasharray="4 4" dot={false} />
      </LineChart>
    </ResponsiveContainer>
  );
}

function ConfusionCard({ matrix = null, labels = null }) {
  if (!matrix || !Array.isArray(matrix)) {
    return (
      <Typography variant="body2" color="textSecondary">
        No confusion matrix available. If you want to demonstrate classification metrics, pass `report.confusion_matrix` from the server in format [[tn, fp],[fn, tp]] or a per-class matrix.
      </Typography>
    );
  }

  if (matrix.length === 2 && matrix[0].length === 2) {
    const tn = matrix[0][0];
    const fp = matrix[0][1];
    const fn = matrix[1][0];
    const tp = matrix[1][1];
    const total = tn + fp + fn + tp || 1;
    const labelsLocal = labels || ["Neg", "Pos"];
    return (
      <Table size="small">
        <TableBody>
          <TableRow>
            <TableCell />
            <TableCell align="center"><strong>Pred: {labelsLocal[0]}</strong></TableCell>
            <TableCell align="center"><strong>Pred: {labelsLocal[1]}</strong></TableCell>
            <TableCell />
          </TableRow>
          <TableRow>
            <TableCell component="th" scope="row"><strong>Actual: {labelsLocal[0]}</strong></TableCell>
            <TableCell align="center">{tn} ({Math.round((tn / total) * 100)}%)</TableCell>
            <TableCell align="center">{fp} ({Math.round((fp / total) * 100)}%)</TableCell>
            <TableCell component="th" scope="row">TN / FP</TableCell>
          </TableRow>
          <TableRow>
            <TableCell component="th" scope="row"><strong>Actual: {labelsLocal[1]}</strong></TableCell>
            <TableCell align="center">{fn} ({Math.round((fn / total) * 100)}%)</TableCell>
            <TableCell align="center">{tp} ({Math.round((tp / total) * 100)}%)</TableCell>
            <TableCell component="th" scope="row">FN / TP</TableCell>
          </TableRow>
          <TableRow>
            <TableCell colSpan={4}>
              <Typography variant="caption">Total: {total} items</Typography>
            </TableCell>
          </TableRow>
        </TableBody>
      </Table>
    );
  }

  return (
    <pre style={{ whiteSpace: "pre-wrap", margin: 0 }}>{JSON.stringify(matrix, null, 2)}</pre>
  );
}

// ----------------- ACE modal -----------------
function AceModal({ open, ace, onClose }) {
  return (
    <Dialog open={!!open} onClose={onClose} maxWidth="md" fullWidth>
      <DialogTitle>ACE JSON {ace?.ace_id ? ` — ${ace.ace_id}` : ""}</DialogTitle>
      <DialogContent dividers>
        <pre style={{ whiteSpace: "pre-wrap", fontSize: 12 }}>{JSON.stringify(ace || {}, null, 2)}</pre>
      </DialogContent>
      <DialogActions>
        <Button onClick={onClose} variant="outlined">Close</Button>
      </DialogActions>
    </Dialog>
  );
}

// ----------------- bar label -----------------
function renderBarLabel(props) {
  const { x, y, width, value } = props;
  const pct = Math.round((value ?? 0) * 100);
  return (
    <text x={x + width / 2} y={y - 6} textAnchor="middle" fontSize={12} fill="#222">
      {pct}%
    </text>
  );
}

// ----------------- main app -----------------
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
  const [batchProgress, setBatchProgress] = useState(null);
  const latestReportRef = useRef(null);

  // ACE modal state
  const [aceModalOpen, setAceModalOpen] = useState(false);
  const [aceModalItem, setAceModalItem] = useState(null);
  function openAceModal(item) {
    setAceModalItem(item);
    setAceModalOpen(true);
  }
  function closeAceModal() {
    setAceModalItem(null);
    setAceModalOpen(false);
  }

  // Export menu state
  const [exportAnchor, setExportAnchor] = useState(null);
  const openExportMenu = (e) => setExportAnchor(e.currentTarget);
  const closeExportMenu = () => setExportAnchor(null);

  // load datasets on start
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

  // load files when dataset changes
  useEffect(() => {
    if (!dataset) return;
    (async () => {
      try {
        const data = await fetchFiles(dataset);
        let parsed = parseMaybeJson(data, { samples: [], count: 0 });
        if (typeof parsed === "string") {
          try {
            parsed = JSON.parse(parsed);
          } catch (e) {
            try {
              const once = JSON.parse(parsed);
              if (typeof once === "string") parsed = JSON.parse(once);
              else parsed = once;
            } catch {
              parsed = { samples: [], count: 0 };
            }
          }
        }

        const samples = Array.isArray(parsed.samples) ? parsed.samples : [];
        setAvailableFiles(samples);
        setFileCount(Number(parsed.count || samples.length || 0));

        if (samples.length >= 2) {
          const v1 = samples.find((s) => /-v1\./i.test(s)) || samples[0];
          const v2 =
            samples.find((s) => /-v2\./i.test(s) && s !== v1) ||
            samples.find((s) => s !== v1) ||
            samples[0];
          setOld(v1);
          setNew(v2);
        } else if (samples.length === 1) {
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

  // run analysis
  const run = async (opts = {}) => {
    setErr("");
    setReport(null);
    setLoading(true);
    const usePairId = opts.usePair ?? !!pairId;
    try {
      const data = await analyzeAPI(oldSpec, newSpec, dataset, usePairId ? pairId : null);
      const parsed = parseMaybeJson(data, null) || {};
      // add small metadata to help exports
      parsed.metadata = parsed.metadata || {};
      if (!parsed.metadata.dataset) parsed.metadata.dataset = dataset;
      if (!parsed.metadata.file_name) parsed.metadata.file_name = newSpec || "";
      if (!parsed.metadata.generated_at) parsed.metadata.generated_at = new Date().toISOString();
      parsed.metadata.commit_hash = parsed.metadata.commit_hash || parsed.versioning?.commit || parsed.versioning?.git_commit || null;
      parsed.metadata.repo_url = parsed.metadata.repo_url || (parsed.metadata.repo_owner && parsed.metadata.repo_name ? `https://github.com/${parsed.metadata.repo_owner}/${parsed.metadata.repo_name}` : parsed.metadata.repo_url || null);

      setReport(parsed);
      latestReportRef.current = parsed;
      setLastRequest({ oldSpec, newSpec, dataset, pairId: usePairId ? pairId : null });

      // try to auto-fetch versioning if logs contain pair_id
      if (parsed && parsed.versioning && Object.keys(parsed.versioning).length === 0 && parsed.logs) {
        const logEntry = (parsed.logs || []).find((l) => typeof l === "string" && l.includes("pair_id="));
        if (logEntry) {
          const m = /pair_id=([^\s,;]+)/.exec(logEntry);
          if (m) {
            const pid = m[1];
            setPairId(pid);
            try {
              const v = await fetchVersioning(pid);
              setReport((r) => ({ ...r, versioning: v }));
            } catch {}
          }
        }
      }

      // fetch example consumers
      if (parsed && parsed.backend_impacts && parsed.backend_impacts.length > 0) {
        try {
          const svc = parsed.backend_impacts[0].service;
          const c = await fetchConsumers(svc);
          setConsumers(parseMaybeJson(c, []));
        } catch {
          setConsumers([]);
        }
      } else {
        setConsumers([]);
      }
    } catch (e) {
      setErr(e?.message || "Failed to analyze. Check backend logs.");
    } finally {
      setLoading(false);
    }
  };

  // retry last request
  const retry = async () => {
    if (!lastRequest) return;
    setErr("");
    setLoading(true);
    try {
      const r = await analyzeAPI(lastRequest.oldSpec, lastRequest.newSpec, lastRequest.dataset, lastRequest.pairId);
      const parsed = parseMaybeJson(r, null) || {};
      parsed.metadata = parsed.metadata || {};
      if (!parsed.metadata.generated_at) parsed.metadata.generated_at = new Date().toISOString();
      setReport(parsed);
      latestReportRef.current = parsed;
    } catch (e) {
      setErr(e?.message || "Retry failed");
    } finally {
      setLoading(false);
    }
  };

  // export helpers
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

  const exportAceJson = (ace) => {
    const name = `ace-${ace?.ace_id || "unknown"}-${Date.now()}.json`;
    downloadJSON(ace || {}, name);
  };

  // batch runner
  const runBatch = async (count = 5) => {
    if (!dataset) return;
    setBatchProgress({ running: true, done: 0, total: count });
    try {
      const results = await fetchReportsBatch({
        dataset,
        limit: count,
        onProgress: (i, total, meta) => {
          setBatchProgress({ running: true, done: i, total });
        },
      });
      const ndjson = results.map((r) => JSON.stringify(r)).join("\n");
      const blob = new Blob([ndjson], { type: "application/x-ndjson" });
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = `batch-reports-${dataset}-${Date.now()}.ndjson`;
      a.click();
      URL.revokeObjectURL(url);
      setBatchProgress({ running: false, done: results.length, total: results.length });
    } catch (e) {
      setErr("Batch run failed: " + (e?.message || e));
      setBatchProgress({ running: false, done: 0, total: count });
    }
  };

  // risk helper
  const riskLevel = (score) => {
    const s = score ?? 0;
    if (s >= 0.7) return { label: "High", color: "error" };
    if (s >= 0.4) return { label: "Medium", color: "warning" };
    return { label: "Low", color: "success" };
  };

  const topB = toTop(report?.backend_impacts);
  const topF = toTop(report?.frontend_impacts);

  // render explanation
  const renderHumanExplanation = (rpt) => {
    if (!rpt) return null;
    const score = rpt.predicted_risk ?? rpt.risk_score ?? 0;
    const band = riskLevel(score).label;
    const numChanges = (rpt.details && rpt.details.length) || 0;
    const backendCount = (rpt.backend_impacts && rpt.backend_impacts.length) || 0;
    const frontendCount = (rpt.frontend_impacts && rpt.frontend_impacts.length) || 0;
    const confOverall = Math.round((rpt.confidence?.overall ?? 0) * 100);
    const confBackend = Math.round((rpt.confidence?.backend ?? 0) * 100);
    const confFrontend = Math.round((rpt.confidence?.frontend ?? 0) * 100);

    const bullets = [
      `Risk: ${band} (${(score).toFixed(2)}) — confidence overall ${confOverall}%`,
      `${numChanges} API change${numChanges !== 1 ? "s" : ""} detected`,
      `${backendCount} backend module${backendCount !== 1 ? "s" : ""} potentially impacted`,
      `${frontendCount} frontend module${frontendCount !== 1 ? "s" : ""} potentially impacted`,
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
            const pathOrDetail = d.detail || d.path || "";
            const aceId = d.ace_id ? ` [${d.ace_id}]` : "";
            const reason = explainChange(d, rpt);
            return (
              <li key={i} style={{ marginBottom: 8 }}>
                <Typography variant="body2">
                  <strong
                    style={{ cursor: "pointer", color: "#1565c0" }}
                    onClick={() => openAceModal(d)}
                  >
                    {human}
                  </strong>{aceId} — {pathOrDetail}
                </Typography>
                <Typography variant="body2" style={{ color: "#555", marginTop: 4 }}>
                  {reason}
                  <span style={{ marginLeft: 8 }}>
                    <Button size="small" variant="text" onClick={() => openAceModal(d)}>View ACE</Button>
                    <Button size="small" variant="text" onClick={() => exportAceJson(d)}>Export ACE</Button>
                  </span>
                </Typography>
              </li>
            );
          })}
        </ol>
      </>
    );
  };

  const computed = computeSummaryFromReport(report || {});
  const acesCount = computed.aces;
  const backendCount = computed.backendImpactsCount;
  const frontendCount = computed.frontendImpactsCount;
  const calibrationBins = computed.calibration_bins;
  const confusionMatrix = computed.confusion_matrix;
  const confusionLabels = computed.confusion_labels;

  // build repo link if available
  const repoLink = (r) => {
    if (!r) return null;
    if (r.metadata?.repo_url) return r.metadata.repo_url;
    if (r.metadata?.repo_owner && r.metadata?.repo_name) return `https://github.com/${r.metadata.repo_owner}/${r.metadata.repo_name}`;
    return null;
  };

  return (
    <Container maxWidth="lg" style={{ padding: "1rem 0 2rem 0" }}>
      <Typography variant="h5" align="center" gutterBottom>
        Impact AI — Prototype Dashboard {dataset ? `| ${dataset.toUpperCase()}` : ""}
      </Typography>

      {/* Controls */}
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
                  setOld(e.target.value);
                  if (autoSync) setNew(e.target.value);
                }}
                SelectProps={{ MenuProps: { PaperProps: { style: { maxHeight: 360 } } } }}
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
              <TextField size="small" fullWidth label="Pair ID (optional)" value={pairId} onChange={(e) => setPairId(e.target.value)} />
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

            <Grid item xs={12}>
              <FormControlLabel control={<Switch checked={autoSync} onChange={(e) => setAutoSync(e.target.checked)} />} label="Auto sync new = old when selecting" />
            </Grid>

            <Grid item xs={12}>
              <Stack direction="row" spacing={2}>
                <Button variant="outlined" onClick={() => runBatch(10)} startIcon={<PlayArrowIcon />}>
                  Batch: 10
                </Button>
                <Button variant="outlined" onClick={() => runBatch(25)} startIcon={<PlayArrowIcon />}>
                  Batch: 25
                </Button>
                <Button
                  variant="contained"
                  color="primary"
                  onClick={async () => {
                    await trainModel([{ change: "sample" }]);
                    alert("train submitted");
                  }}
                >
                  Send Sample
                </Button>
              </Stack>
            </Grid>
          </Grid>
        </CardContent>
      </Card>

      {/* Error */}
      {err && (
        <Card style={{ background: "#fff3f0", marginBottom: 12 }}>
          <CardContent>
            <Typography color="error">{String(err)}</Typography>
          </CardContent>
        </Card>
      )}

      {/* Batch progress */}
      {batchProgress && (
        <Card style={{ marginBottom: 12 }}>
          <CardContent>
            <Typography variant="subtitle2">Batch run progress</Typography>
            <LinearProgress variant="determinate" value={(batchProgress.done / (batchProgress.total || 1)) * 100} />
            <Typography variant="caption">
              {batchProgress.done}/{batchProgress.total}
            </Typography>
          </CardContent>
        </Card>
      )}

      {/* Report card */}
      {report && (
        <>
          {/* Provenance banner */}
          <Box marginBottom={2}>
            <Card style={{ background: "#f6f7f9" }}>
              <CardContent>
                <Grid container alignItems="center" spacing={2}>
                  <Grid item xs={12} md={7}>
                    <Typography variant="subtitle2">
                      {report.logs && report.logs.find((l) => l.includes("pair_id=")) ? (
                        <>
                          Curated pair detected — <strong>{(report.logs.find((l) => l.includes("pair_id=")) || "").split("pair_id=")[1]}</strong>
                        </>
                      ) : report.versioning && Object.keys(report.versioning).length > 0 ? (
                        <>
                          Version metadata available — <strong>{report.versioning.semver_old || report.versioning.semver_new || "meta"}</strong>
                        </>
                      ) : (
                        <>Ad-hoc analysis (no curated pair)</>
                      )}
                    </Typography>
                    <Typography variant="body2" color="textSecondary">
                      {report.logs && report.logs.join(" • ")}
                    </Typography>
                    <Typography variant="caption" style={{ display: "block", marginTop: 6 }}>
                      ACES: {report.summary_counts?.aces ?? acesCount} &nbsp;|&nbsp; Backend hits: {report.summary_counts?.backend_impacts ?? backendCount} &nbsp;|&nbsp; Frontend hits: {report.summary_counts?.frontend_impacts ?? frontendCount}
                    </Typography>

                    {repoLink(report) && (
                      <div style={{ marginTop: 6 }}>
                        Repo: <code style={{ fontSize: 12 }}>
                          <a href={repoLink(report)} target="_blank" rel="noreferrer">{(report.metadata?.repo_url || repoLink(report)).replace(/^https?:\/\//, "")}</a>
                        </code>
                      </div>
                    )}

                    {report.metadata?.commit_hash && (
                      <div style={{ marginTop: 4 }}>
                        Commit: <code style={{ fontSize: 12 }}>{report.metadata.commit_hash.slice(0, 10)}</code>
                      </div>
                    )}

                    {report.metadata?.generated_at && (
                      <div style={{ marginTop: 4 }}>
                        <Typography variant="caption">Generated: {report.metadata.generated_at}</Typography>
                      </div>
                    )}
                  </Grid>
                  <Grid item xs={12} md={5} style={{ textAlign: "right" }}>
                    <Tooltip title="Export menu">
                      <IconButton onClick={openExportMenu}>
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
                      <IconButton onClick={retry}>
                        <ReplayIcon />
                      </IconButton>
                    </Tooltip>
                  </Grid>
                </Grid>
              </CardContent>
            </Card>
          </Box>

          {/* Top row */}
          <Grid container spacing={2} marginBottom={2}>
            <Grid item xs={12} md={4}>
              <Card>
                <CardContent>
                  <Typography variant="subtitle1">Predicted Risk</Typography>
                  <Box position="relative" display="inline-flex" marginTop={1}>
                    <CircularProgress
                      variant="determinate"
                      value={(report.predicted_risk ?? 0) * 100}
                      size={92}
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
                  <Typography variant="body2" style={{ marginTop: 8 }}>
                    {riskLevel(report.predicted_risk).label}
                  </Typography>
                </CardContent>
              </Card>
            </Grid>

            <Grid item xs={12} md={8}>
              <Card>
                <CardContent>
                  <Typography variant="subtitle1">Confidence</Typography>
                  <Typography variant="caption">Overall</Typography>
                  <LinearProgress variant="determinate" value={(report.confidence?.overall ?? 0) * 100} />
                  <Typography variant="caption">Backend</Typography>
                  <LinearProgress variant="determinate" value={(report.confidence?.backend ?? 0) * 100} />
                  <Typography variant="caption">Frontend</Typography>
                  <LinearProgress variant="determinate" value={(report.confidence?.frontend ?? 0) * 100} />
                </CardContent>
              </Card>
            </Grid>
          </Grid>

          {/* charts */}
          <Grid container spacing={2} marginBottom={2}>
            <Grid item xs={12} md={6}>
              <Card>
                <CardContent>
                  <Typography variant="h6">Backend Impacts</Typography>
                  {topB.length === 0 ? (
                    <Typography variant="body2" color="textSecondary">
                      No backend impacts detected.
                    </Typography>
                  ) : (
                    <ResponsiveContainer width="100%" height={240}>
                      <BarChart data={topB} margin={{ top: 6, right: 8, left: 0, bottom: 28 }}>
                        <CartesianGrid strokeDasharray="3 3" />
                        <XAxis dataKey="name" tick={<CustomizedAxisTick />} interval={0} height={70} allowDataOverflow />
                        <YAxis domain={[0, 1]} tickFormatter={(v) => `${Math.round(v * 100)}%`} />
                        <ReTooltip
                          formatter={(value) => `${Math.round(value * 100)}%`}
                          labelFormatter={(label, payload) => (payload && payload[0] && payload[0].payload.fullName) || label}
                        />
                        <Bar dataKey="risk" barSize={26} label={renderBarLabel} isAnimationActive={false} />
                      </BarChart>
                    </ResponsiveContainer>
                  )}
                </CardContent>
              </Card>
            </Grid>

            <Grid item xs={12} md={6}>
              <Card>
                <CardContent>
                  <Typography variant="h6">Frontend Impacts</Typography>
                  {topF.length === 0 ? (
                    <Typography variant="body2" color="textSecondary">
                      No frontend impacts detected.
                    </Typography>
                  ) : (
                    <ResponsiveContainer width="100%" height={240}>
                      <BarChart data={topF} margin={{ top: 6, right: 8, left: 0, bottom: 28 }}>
                        <CartesianGrid strokeDasharray="3 3" />
                        <XAxis dataKey="name" tick={<CustomizedAxisTick />} interval={0} height={70} allowDataOverflow />
                        <YAxis domain={[0, 1]} tickFormatter={(v) => `${Math.round(v * 100)}%`} />
                        <ReTooltip
                          formatter={(value) => `${Math.round(value * 100)}%`}
                          labelFormatter={(label, payload) => (payload && payload[0] && payload[0].payload.fullName) || label}
                        />
                        <Bar dataKey="risk" barSize={26} label={renderBarLabel} isAnimationActive={false} />
                      </BarChart>
                    </ResponsiveContainer>
                  )}
                </CardContent>
              </Card>
            </Grid>
          </Grid>

          {/* Calibration & Confusion */}
          <Grid container spacing={2} marginBottom={2}>
            <Grid item xs={12} md={6}>
              <Card>
                <CardContent>
                  <Typography variant="h6">Calibration</Typography>
                  <Box marginTop={1}>
                    <CalibrationChart bins={calibrationBins} />
                  </Box>
                </CardContent>
              </Card>
            </Grid>

            {confusionMatrix ? (
              <Grid item xs={12} md={6}>
                <Card>
                  <CardContent>
                    <Typography variant="h6">Confusion Matrix</Typography>
                    <Box marginTop={1}>
                      <ConfusionCard matrix={confusionMatrix} labels={confusionLabels} />
                    </Box>
                  </CardContent>
                </Card>
              </Grid>
            ) : (
              <Grid item xs={12} md={6}>
                <Card>
                  <CardContent>
                    <Typography variant="h6">Confusion Matrix</Typography>
                    <Typography variant="body2" color="textSecondary">
                      No confusion matrix available. Hidden by default when no data present.
                    </Typography>
                  </CardContent>
                </Card>
              </Grid>
            )}
          </Grid>

          {/* improved humanized explanation */}
          <Card style={{ marginBottom: 12 }}>
            <CardContent>
              <Typography variant="h6">AI Explanation</Typography>
              <Box marginTop={1}>{renderHumanExplanation(report)}</Box>
            </CardContent>
          </Card>

          {/* versioning (kept for debugging) */}
          {report.versioning && (
            <Card style={{ marginBottom: 12 }}>
              <CardContent>
                <Typography variant="h6">Versioning / Metadata</Typography>
                <pre style={{ whiteSpace: "pre-wrap", margin: 0 }}>{JSON.stringify(report.versioning, null, 2)}</pre>
              </CardContent>
            </Card>
          )}

          {/* consumers */}
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

      {/* ACE modal */}
      <AceModal open={aceModalOpen} ace={aceModalItem} onClose={closeAceModal} />
    </Container>
  );
}
