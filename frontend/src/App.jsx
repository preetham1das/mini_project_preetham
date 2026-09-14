import { useState, useEffect } from "react";
import "./App.css";

/* ── SVG Icons ───────────────────────────────────────────── */
const IconUpload = () => (
  <svg width="28" height="28" fill="none" stroke="currentColor" strokeWidth="1.5" viewBox="0 0 24 24">
    <path d="M21 15v4a2 2 0 01-2 2H5a2 2 0 01-2-2v-4" strokeLinecap="round" strokeLinejoin="round"/>
    <polyline points="17 8 12 3 7 8" strokeLinecap="round" strokeLinejoin="round"/>
    <line x1="12" y1="3" x2="12" y2="15" strokeLinecap="round"/>
  </svg>
);

const IconCheck = () => (
  <svg width="14" height="14" fill="none" stroke="currentColor" strokeWidth="2.5" viewBox="0 0 24 24">
    <polyline points="20 6 9 17 4 12" strokeLinecap="round" strokeLinejoin="round"/>
  </svg>
);

const IconSatellite = () => (
  <svg width="18" height="18" fill="none" stroke="currentColor" strokeWidth="1.8" viewBox="0 0 24 24">
    <path d="M12 2L2 7l10 5 10-5-10-5z" strokeLinejoin="round"/>
    <path d="M2 17l10 5 10-5" strokeLinecap="round" strokeLinejoin="round"/>
    <path d="M2 12l10 5 10-5" strokeLinecap="round" strokeLinejoin="round"/>
  </svg>
);

const IconClock = () => (
  <svg width="18" height="18" fill="none" stroke="currentColor" strokeWidth="1.8" viewBox="0 0 24 24">
    <circle cx="12" cy="12" r="10"/>
    <polyline points="12 6 12 12 16 14" strokeLinecap="round" strokeLinejoin="round"/>
  </svg>
);

const IconSpinner = () => (
  <svg className="spinner-icon" width="18" height="18" viewBox="0 0 24 24" fill="none">
    <circle cx="12" cy="12" r="10" stroke="rgba(255,255,255,0.25)" strokeWidth="2.5"/>
    <path d="M12 2a10 10 0 0110 10" stroke="white" strokeWidth="2.5" strokeLinecap="round"/>
  </svg>
);

/* ── Progress Ring ───────────────────────────────────────── */
const ProgressRing = ({ percent, color, size = 108, stroke = 9 }) => {
  const r = (size - stroke) / 2;
  const circ = 2 * Math.PI * r;
  const offset = circ - (Math.min(percent, 100) / 100) * circ;
  return (
    <svg width={size} height={size} style={{ transform: "rotate(-90deg)" }}>
      <circle cx={size / 2} cy={size / 2} r={r} fill="none" stroke="#e2e8f0" strokeWidth={stroke} />
      <circle
        key={percent}
        cx={size / 2} cy={size / 2} r={r}
        fill="none" stroke={color} strokeWidth={stroke}
        strokeDasharray={circ} strokeDashoffset={offset}
        strokeLinecap="round"
        style={{ transition: "stroke-dashoffset 1.1s cubic-bezier(.4,0,.2,1)" }}
      />
    </svg>
  );
};

/* ── Severity helpers ────────────────────────────────────── */
function forestSeverity(pct) {
  if (pct >= 60) return { color: "#16a34a", bg: "#dcfce7", label: "High Coverage" };
  if (pct >= 30) return { color: "#ca8a04", bg: "#fef9c3", label: "Moderate Coverage" };
  return { color: "#dc2626", bg: "#fee2e2", label: "Low Coverage" };
}
function defoSeverity(pct) {
  if (pct < 5)  return { color: "#16a34a", bg: "#dcfce7", label: "Minimal Change" };
  if (pct < 20) return { color: "#ca8a04", bg: "#fef9c3", label: "Moderate Deforestation" };
  return { color: "#dc2626", bg: "#fee2e2", label: "Severe Deforestation" };
}

/* ══════════════════════════════════════════════════════════ */
function App() {
  const [mode, setMode] = useState("forest");
  const [samples, setSamples] = useState({ stage1_samples: [], stage2_samples: [] });

  // Stage 1 state
  const [file, setFile]           = useState(null);
  const [preview, setPreview]     = useState(null);
  const [samplePath, setSamplePath] = useState("");
  const [thresholdS1, setThresholdS1] = useState(0.5);

  // Stage 2 state
  const [fileT1, setFileT1]         = useState(null);
  const [previewT1, setPreviewT1]   = useState(null);
  const [fileT2, setFileT2]         = useState(null);
  const [previewT2, setPreviewT2]   = useState(null);
  const [samplePair, setSamplePair] = useState("");
  const [thresholdS2, setThresholdS2] = useState(0.5);

  const [result, setResult]   = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError]     = useState(null);

  useEffect(() => {
    fetch("http://127.0.0.1:8000/api/samples")
      .then(r => r.json())
      .then(d => setSamples(d))
      .catch(() => {});
  }, []);

  const resetAll = () => {
    setResult(null); setError(null);
    setFile(null); setPreview(null); setSamplePath("");
    setFileT1(null); setPreviewT1(null);
    setFileT2(null); setPreviewT2(null);
    setSamplePair("");
  };

  const handleModeChange = (m) => { setMode(m); resetAll(); };

  // Stage 1 handlers
  const handleFile = (e) => {
    const f = e.target.files[0]; if (!f) return;
    setFile(f); setPreview(URL.createObjectURL(f));
    setSamplePath(""); setResult(null); setError(null);
  };
  const handleSample1 = (e) => {
    setSamplePath(e.target.value);
    setFile(null); setPreview(null);
    setResult(null); setError(null);
  };

  // Stage 2 handlers
  const handleFileT1 = (e) => {
    const f = e.target.files[0]; if (!f) return;
    setFileT1(f); setPreviewT1(URL.createObjectURL(f));
    setSamplePair(""); setResult(null); setError(null);
  };
  const handleFileT2 = (e) => {
    const f = e.target.files[0]; if (!f) return;
    setFileT2(f); setPreviewT2(URL.createObjectURL(f));
    setSamplePair(""); setResult(null); setError(null);
  };
  const handleSample2 = (e) => {
    setSamplePair(e.target.value);
    setFileT1(null); setPreviewT1(null);
    setFileT2(null); setPreviewT2(null);
    setResult(null); setError(null);
  };

  // API calls
  const analyzeImage = async () => {
    if (!file && !samplePath) return;
    setLoading(true); setResult(null); setError(null);
    const fd = new FormData();
    if (file) fd.append("file", file);
    if (samplePath) fd.append("sample_path", samplePath);
    fd.append("threshold", thresholdS1);
    try {
      const res = await fetch("http://127.0.0.1:8000/predict", { method: "POST", body: fd });
      const data = await res.json();
      if (!res.ok || !data.success) throw new Error(data.error || "Prediction failed");
      setResult(data);
    } catch (err) { setError(err.message); }
    finally { setLoading(false); }
  };

  const analyzeDeforestation = async () => {
    if ((!fileT1 || !fileT2) && !samplePair) return;
    setLoading(true); setResult(null); setError(null);
    const fd = new FormData();
    if (fileT1 && fileT2) {
      fd.append("before_image", fileT1);
      fd.append("after_image", fileT2);
    }
    if (samplePair) {
      const obj = JSON.parse(samplePair);
      fd.append("sample_path_a", obj.path_a);
      fd.append("sample_path_b", obj.path_b);
    }
    fd.append("threshold", thresholdS2);
    try {
      const res = await fetch("http://127.0.0.1:8000/predict-temporal", { method: "POST", body: fd });
      const data = await res.json();
      if (!res.ok || !data.success) throw new Error(data.error || "Prediction failed");
      setResult(data);
    } catch (err) { setError(err.message); }
    finally { setLoading(false); }
  };

  const canRunS1 = !!(file || samplePath);
  const canRunS2 = !!((fileT1 && fileT2) || samplePair);

  const fSev = result?.forest_percentage != null ? forestSeverity(result.forest_percentage) : null;
  const dSev = result?.deforestation_percentage != null ? defoSeverity(result.deforestation_percentage) : null;

  return (
    <div className="app">

      {/* ── HEADER ─────────────────────────────────────────── */}
      <header className="header">
        <div className="header-inner">
          <div className="header-brand">
            <div className="header-logo">🌿</div>
            <div className="header-text">
              <h1 className="header-title">ForestWatch AI</h1>
              <p className="header-sub">
                Forest Cover Segmentation &amp; Temporal Deforestation Detection · U-Net Architecture
              </p>
            </div>
          </div>
          <div className="header-badges">
            <span className="hbadge">Stage 1 · Segmentation</span>
            <span className="hbadge">Stage 2 · Change Detection</span>
            <span className="hbadge hbadge--live">● Live</span>
          </div>
        </div>
      </header>

      <main className="main">

        {/* ── MODE TABS ──────────────────────────────────────── */}
        <div className="tab-bar">
          <button
            className={`tab-btn${mode === "forest" ? " tab-btn--active" : ""}`}
            onClick={() => handleModeChange("forest")}
          >
            <IconSatellite />
            <span>Forest Cover Detection</span>
            <span className="tab-pill tab-pill--green">Stage 1</span>
          </button>
          <button
            className={`tab-btn${mode === "temporal" ? " tab-btn--active" : ""}`}
            onClick={() => handleModeChange("temporal")}
          >
            <IconClock />
            <span>Deforestation Detection</span>
            <span className="tab-pill tab-pill--amber">Stage 2</span>
          </button>
        </div>

        {/* ── ERROR BANNER ───────────────────────────────────── */}
        {error && (
          <div className="error-banner fade-in">
            <span>⚠️ &nbsp;{error}</span>
            <button className="error-dismiss" onClick={() => setError(null)}>✕</button>
          </div>
        )}

        {/* ══════════════ STAGE 1 ══════════════════════════════ */}
        {mode === "forest" && (
          <div className="fade-in">
            <p className="mode-desc">
              Upload a satellite image to segment forest pixels using a <strong>U-Net</strong> trained on the Amazon Forest dataset.
            </p>

            <div className="panel">
              <div className="panel-head">
                <span className="panel-title">Input Image</span>
                <span className="panel-hint">Pick a sample or upload your own</span>
              </div>

              {samples.stage1_samples.length > 0 && (
                <div className="field">
                  <label className="field-label">Sample Library</label>
                  <select className="styled-select" value={samplePath} onChange={handleSample1}>
                    <option value="">— select a preset sample image —</option>
                    {samples.stage1_samples.map(s => (
                      <option key={s.path} value={s.path}>[{s.split}] {s.name}</option>
                    ))}
                  </select>
                </div>
              )}

              <div className="or-divider"><span>or upload custom image</span></div>

              <label className={`drop-zone${file ? " drop-zone--filled" : ""}`}>
                <input type="file" accept="image/*,.tif,.tiff" onChange={handleFile} />
                <div className="drop-zone-body">
                  <IconUpload />
                  {file ? (
                    <>
                      <span className="drop-filename"><IconCheck /> {file.name}</span>
                      <span className="drop-hint">Click to replace</span>
                    </>
                  ) : (
                    <>
                      <span className="drop-label">Click to upload satellite image</span>
                      <span className="drop-hint">PNG · JPEG · TIFF supported</span>
                    </>
                  )}
                </div>
              </label>

              <div className="threshold-box">
                <div className="threshold-top">
                  <label className="field-label">Classification Threshold</label>
                  <span className="threshold-chip">{thresholdS1}</span>
                </div>
                <input
                  type="range" className="range-input"
                  min="0.1" max="0.9" step="0.05"
                  value={thresholdS1}
                  onChange={e => setThresholdS1(parseFloat(e.target.value))}
                />
                <div className="threshold-labels">
                  <span>Conservative</span><span>Balanced</span><span>Strict</span>
                </div>
              </div>

              <div className="panel-action">
                <button
                  className={`run-btn run-btn--green${!canRunS1 || loading ? " run-btn--off" : ""}`}
                  onClick={analyzeImage}
                  disabled={!canRunS1 || loading}
                >
                  {loading
                    ? <><IconSpinner /> Analyzing…</>
                    : <>▶ &nbsp;Analyze Forest Cover</>
                  }
                </button>
              </div>
            </div>

            {/* Preview strip */}
            {preview && !result && (
              <div className="preview-row fade-in">
                <div className="preview-card">
                  <div className="preview-card-label">Selected Image</div>
                  <img src={preview} alt="Preview" />
                </div>
              </div>
            )}

            {/* Results */}
            {result && (
              <div className="results fade-in">
                <div className="results-head">
                  <h2 className="results-title">Analysis Results</h2>
                  <button className="ghost-btn" onClick={() => setResult(null)}>Clear</button>
                </div>

                <div className="img-grid img-grid--two">
                  <div className="img-card">
                    <div className="img-card-head">
                      <span className="img-card-label">Original Image</span>
                      <span className="img-card-meta">{result.original_width}×{result.original_height}px</span>
                    </div>
                    <img src={`data:image/png;base64,${result.original_image}`} alt="Satellite" />
                  </div>
                  <div className="img-card">
                    <div className="img-card-head">
                      <span className="img-card-label">Forest Segmentation</span>
                      <span className="badge-pill badge-pill--green">U-Net Mask</span>
                    </div>
                    <img src={`data:image/png;base64,${result.mask}`} alt="Forest Mask" />
                  </div>
                </div>

                <div className="metric-card">
                  <div className="metric-ring-wrap">
                    <ProgressRing
                      key={result.forest_percentage}
                      percent={result.forest_percentage}
                      color={fSev.color}
                    />
                    <div className="metric-ring-label" style={{ color: fSev.color }}>
                      {result.forest_percentage}%
                    </div>
                  </div>
                  <div className="metric-body">
                    <h3 className="metric-title">Forest Coverage</h3>
                    <span
                      className="severity-pill"
                      style={{ background: fSev.bg, color: fSev.color }}
                    >
                      {fSev.label}
                    </span>
                    <div className="metric-stats">
                      <div className="stat">
                        <span>{result.forest_pixels.toLocaleString()}</span>
                        <label>Forest Pixels</label>
                      </div>
                      <div className="stat">
                        <span>{result.total_pixels.toLocaleString()}</span>
                        <label>Total Pixels</label>
                      </div>
                      <div className="stat">
                        <span>{result.threshold}</span>
                        <label>Threshold</label>
                      </div>
                    </div>
                  </div>
                </div>
              </div>
            )}
          </div>
        )}

        {/* ══════════════ STAGE 2 ══════════════════════════════ */}
        {mode === "temporal" && (
          <div className="fade-in">
            <p className="mode-desc">
              Upload a <strong>before</strong> and <strong>after</strong> image pair to detect deforestation using a
              <strong> 9-channel Temporal U-Net</strong> trained on the Forest-Change dataset.
            </p>

            <div className="panel">
              <div className="panel-head">
                <span className="panel-title">Bi-Temporal Image Pair</span>
                <span className="panel-hint">Upload both images or select a sample pair</span>
              </div>

              {samples.stage2_samples.length > 0 && (
                <div className="field">
                  <label className="field-label">Sample Pair Library</label>
                  <select className="styled-select" value={samplePair} onChange={handleSample2}>
                    <option value="">— select a preset sample pair —</option>
                    {samples.stage2_samples.map(s => (
                      <option key={s.id} value={JSON.stringify(s)}>[{s.split}] {s.name}</option>
                    ))}
                  </select>
                </div>
              )}

              <div className="or-divider"><span>or upload custom image pair</span></div>

              <div className="temporal-row">
                <div className="temporal-slot">
                  <div className="temporal-tag temporal-tag--t1">T1 · Before</div>
                  <label className={`drop-zone drop-zone--sm${fileT1 ? " drop-zone--filled" : ""}`}>
                    <input type="file" accept="image/*,.tif,.tiff" onChange={handleFileT1} />
                    <div className="drop-zone-body">
                      <IconUpload />
                      {fileT1 ? (
                        <>
                          <span className="drop-filename"><IconCheck /> {fileT1.name}</span>
                          <span className="drop-hint">Click to replace</span>
                        </>
                      ) : (
                        <>
                          <span className="drop-label">Before image (T1)</span>
                          <span className="drop-hint">Earlier acquisition date</span>
                        </>
                      )}
                    </div>
                  </label>
                </div>

                <div className="temporal-arrow">→</div>

                <div className="temporal-slot">
                  <div className="temporal-tag temporal-tag--t2">T2 · After</div>
                  <label className={`drop-zone drop-zone--sm${fileT2 ? " drop-zone--filled" : ""}`}>
                    <input type="file" accept="image/*,.tif,.tiff" onChange={handleFileT2} />
                    <div className="drop-zone-body">
                      <IconUpload />
                      {fileT2 ? (
                        <>
                          <span className="drop-filename"><IconCheck /> {fileT2.name}</span>
                          <span className="drop-hint">Click to replace</span>
                        </>
                      ) : (
                        <>
                          <span className="drop-label">After image (T2)</span>
                          <span className="drop-hint">Later acquisition date</span>
                        </>
                      )}
                    </div>
                  </label>
                </div>
              </div>

              <div className="threshold-box">
                <div className="threshold-top">
                  <label className="field-label">Detection Threshold</label>
                  <span className="threshold-chip">{thresholdS2}</span>
                </div>
                <input
                  type="range" className="range-input"
                  min="0.1" max="0.9" step="0.05"
                  value={thresholdS2}
                  onChange={e => setThresholdS2(parseFloat(e.target.value))}
                />
                <div className="threshold-labels">
                  <span>Sensitive</span><span>Balanced</span><span>Precise</span>
                </div>
              </div>

              <div className="panel-action">
                <button
                  className={`run-btn run-btn--blue${!canRunS2 || loading ? " run-btn--off" : ""}`}
                  onClick={analyzeDeforestation}
                  disabled={!canRunS2 || loading}
                >
                  {loading
                    ? <><IconSpinner /> Analyzing…</>
                    : <>▶ &nbsp;Detect Deforestation</>
                  }
                </button>
              </div>
            </div>

            {/* Preview strip before result */}
            {(previewT1 || previewT2) && !result && (
              <div className="preview-row preview-row--dual fade-in">
                {previewT1 && (
                  <div className="preview-card">
                    <div className="preview-card-label preview-card-label--t1">T1 · Before</div>
                    <img src={previewT1} alt="T1 Preview" />
                  </div>
                )}
                {previewT2 && (
                  <div className="preview-card">
                    <div className="preview-card-label preview-card-label--t2">T2 · After</div>
                    <img src={previewT2} alt="T2 Preview" />
                  </div>
                )}
              </div>
            )}

            {/* Results */}
            {result && (
              <div className="results fade-in">
                <div className="results-head">
                  <h2 className="results-title">Deforestation Analysis</h2>
                  <button className="ghost-btn" onClick={() => setResult(null)}>Clear</button>
                </div>

                <div className="img-grid img-grid--three">
                  <div className="img-card">
                    <div className="img-card-head">
                      <span className="img-card-label">Before (T1)</span>
                      <span className="badge-pill badge-pill--blue">Earlier</span>
                    </div>
                    <img src={`data:image/png;base64,${result.before_image}`} alt="Before" />
                  </div>
                  <div className="img-card">
                    <div className="img-card-head">
                      <span className="img-card-label">After (T2)</span>
                      <span className="badge-pill badge-pill--amber">Later</span>
                    </div>
                    <img src={`data:image/png;base64,${result.after_image}`} alt="After" />
                  </div>
                  <div className="img-card">
                    <div className="img-card-head">
                      <span className="img-card-label">Deforestation Mask</span>
                      <span className="badge-pill badge-pill--purple">9-ch U-Net</span>
                    </div>
                    <img src={`data:image/png;base64,${result.deforestation_mask}`} alt="Mask" />
                  </div>
                </div>

                <div className="metric-card">
                  <div className="metric-ring-wrap">
                    <ProgressRing
                      key={result.deforestation_percentage}
                      percent={result.deforestation_percentage}
                      color={dSev.color}
                    />
                    <div className="metric-ring-label" style={{ color: dSev.color }}>
                      {result.deforestation_percentage}%
                    </div>
                  </div>
                  <div className="metric-body">
                    <h3 className="metric-title">Deforestation Extent</h3>
                    <span
                      className="severity-pill"
                      style={{ background: dSev.bg, color: dSev.color }}
                    >
                      {dSev.label}
                    </span>
                    <div className="metric-stats">
                      <div className="stat">
                        <span>{result.changed_pixels.toLocaleString()}</span>
                        <label>Changed Pixels</label>
                      </div>
                      <div className="stat">
                        <span>{result.total_pixels.toLocaleString()}</span>
                        <label>Total Pixels</label>
                      </div>
                      <div className="stat">
                        <span>{thresholdS2}</span>
                        <label>Threshold</label>
                      </div>
                    </div>
                  </div>
                </div>
              </div>
            )}
          </div>
        )}
      </main>

      <footer className="footer">
        <p>
          AI-Based Forest Cover Segmentation &amp; Temporal Deforestation Detection · U-Net ·
          Final Year Project
        </p>
      </footer>
    </div>
  );
}

export default App;
