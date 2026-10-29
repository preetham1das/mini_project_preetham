import React, { useState, useEffect, useCallback } from 'react';
import {
  Globe,
  Trees,
  Layers,
  GitCompare,
  BarChart3,
  Play,
  Upload,
  MapPin,
  Radar,
  ScanLine,
  Satellite,
  RefreshCw,
  Cpu,
  Activity,
  ShieldCheck,
  ChevronRight,
} from 'lucide-react';
import InteractiveMap from './InteractiveMap.jsx';
import EvaluationPanel from './EvaluationPanel.jsx';

/* Benchmark figures from the recorded evaluation artefacts (outputs/).
   Stage 1 → validation IoU; Stage 2 → best validation IoU during training. */
const FALLBACK_S1_IOU = 0.8386;
const S2_VAL_IOU = 0.5043;

const MANUAL_TABS = [
  { id: 'stage1', label: 'Stage 1 · Canopy', icon: Trees },
  { id: 'stage2', label: 'Stage 2 · Change', icon: Layers },
  { id: 'timeline', label: 'Timeline', icon: GitCompare },
  { id: 'benchmarks', label: 'Benchmarks', icon: BarChart3 },
];

export default function App() {
  const [view, setView] = useState('map'); // 'map' | 'manual'
  const [manualTab, setManualTab] = useState('stage1');

  const [loading, setLoading] = useState(false);
  const [backendStatus, setBackendStatus] = useState('connecting');
  const [sentinelStatus, setSentinelStatus] = useState('checking');

  // Samples
  const [samples, setSamples] = useState({ stage1_samples: [], stage2_samples: [] });

  // GIS / AI overlay state (passed into the map)
  const [gisMode, setGisMode] = useState('stage1');
  const [selectedGisSample, setSelectedGisSample] = useState('');
  const [gisThreshold, setGisThreshold] = useState(0.5);
  const [gisOpacity, setGisOpacity] = useState(0.8);
  const [aiOverlayData, setAiOverlayData] = useState(null);

  // Stage 1 state
  const [s1Sample, setS1Sample] = useState('');
  const [s1File, setS1File] = useState(null);
  const [s1Threshold, setS1Threshold] = useState(0.5);
  const [s1Results, setS1Results] = useState(null);

  // Stage 2 state
  const [s2Sample, setS2Sample] = useState('');
  const [s2FileT1, setS2FileT1] = useState(null);
  const [s2FileT2, setS2FileT2] = useState(null);
  const [s2Threshold, setS2Threshold] = useState(0.5);
  const [s2Results, setS2Results] = useState(null);

  // Timeline state
  const [swipeSample, setSwipeSample] = useState('');
  const [swipeData, setSwipeData] = useState(null);
  const [timelineStep, setTimelineStep] = useState(0);

  // Benchmarks
  const [analytics, setAnalytics] = useState(null);

  useEffect(() => {
    fetch('/api/samples')
      .then((res) => res.json())
      .then((data) => {
        setSamples(data);
        setBackendStatus('online');
      })
      .catch((err) => {
        console.error('Backend error:', err);
        setBackendStatus('offline');
      });

    fetch('/api/metrics')
      .then((res) => res.json())
      .then((data) => setAnalytics(data))
      .catch((err) => console.error('Metrics fetch error:', err));

    fetch('/api/sentinel/status')
      .then((res) => {
        if (!res.ok) throw new Error(`Sentinel status returned ${res.status}`);
        return res.json();
      })
      .then((data) => setSentinelStatus(data.configured ? 'configured' : 'missing'))
      .catch((err) => {
        console.error('Sentinel status error:', err);
        setSentinelStatus('unknown');
      });
  }, []);

  const handleLiveStage2Result = useCallback((data) => {
    const deforestationPct = data.deforestation_pct ?? data.deforestation_percentage;
    const deforestationPixels = data.defor_pixels ?? data.changed_pixels;
    setS2Results({
      ...data,
      pair_name: `Sentinel-2 ${data.date_before || 'T1'} → ${data.date_after || 'T2'}`,
      deforestation_pct: deforestationPct,
      deforestation_pixels: deforestationPixels,
      stable_pixels: data.stable_pixels ?? (data.total_pixels - deforestationPixels),
      stable_pct: data.stable_pct ?? (100 - deforestationPct),
      t1_image: data.before_image ?? data.t1_image,
      t2_image: data.after_image ?? data.t2_image,
      live_source: true,
    });
    // Stay on the map so the new overlay is visible; the manual Stage 2 tab
    // already holds the same result if the user switches over.
    setManualTab('stage2');
  }, []);

  /* Run the model on a pre-loaded sample tile and overlay the mask on the map. */
  const handleRunGis = () => {
    if (!selectedGisSample) {
      alert('Please select a sample satellite tile from the dropdown.');
      return;
    }

    const sampleObj = JSON.parse(selectedGisSample);
    const formData = new FormData();
    formData.append('threshold', gisThreshold);
    setLoading(true);

    const endpoint = gisMode === 'stage1' ? '/api/predict/stage1' : '/api/predict/stage2';
    if (gisMode === 'stage1') {
      formData.append('sample_path', sampleObj.path);
    } else {
      formData.append('sample_id', sampleObj.id);
      formData.append('split', sampleObj.split);
    }

    fetch(endpoint, { method: 'POST', body: formData })
      .then((res) => res.json())
      .then((data) => {
        setLoading(false);
        if (data.error) {
          alert(data.error);
          return;
        }
        const overlayImg = gisMode === 'stage1' ? data.overlay_image : data.deforestation_overlay;
        setAiOverlayData({ image: overlayImg, gis: data.gis, mode: gisMode, metrics: data });
      })
      .catch((err) => {
        setLoading(false);
        console.error(err);
        alert('Failed to connect to backend server!');
      });
  };

  const handleRunStage1 = () => {
    const formData = new FormData();
    formData.append('threshold', s1Threshold);

    if (s1File) {
      formData.append('file', s1File);
    } else if (s1Sample) {
      formData.append('sample_path', s1Sample);
    } else {
      alert('Please select a sample or upload a file.');
      return;
    }

    setLoading(true);
    fetch('/api/predict/stage1', { method: 'POST', body: formData })
      .then((res) => res.json())
      .then((data) => {
        setLoading(false);
        if (data.error) {
          alert(data.error);
          return;
        }
        setS1Results(data);
      })
      .catch(() => {
        setLoading(false);
        alert('Backend connection error.');
      });
  };

  const handleRunStage2 = () => {
    const formData = new FormData();
    formData.append('threshold', s2Threshold);

    if (s2FileT1 && s2FileT2) {
      formData.append('file_t1', s2FileT1);
      formData.append('file_t2', s2FileT2);
    } else if (s2Sample) {
      const sampleObj = JSON.parse(s2Sample);
      formData.append('sample_id', sampleObj.id);
      formData.append('split', sampleObj.split);
    } else {
      if (s2FileT1 && !s2FileT2) {
        alert('Please also upload the After Image (T2).');
      } else if (!s2FileT1 && s2FileT2) {
        alert('Please also upload the Before Image (T1).');
      } else {
        alert('Please select a sample pair or upload both Before (T1) and After (T2) images.');
      }
      return;
    }

    setLoading(true);
    fetch('/api/predict/stage2', { method: 'POST', body: formData })
      .then((res) => res.json())
      .then((data) => {
        setLoading(false);
        if (data.error) {
          alert(data.error);
          return;
        }
        setS2Results(data);
      })
      .catch(() => {
        setLoading(false);
        alert('Backend connection error.');
      });
  };

  const handleLoadSwipe = () => {
    if (!swipeSample) {
      alert('Please select a temporal pair.');
      return;
    }
    const sampleObj = JSON.parse(swipeSample);
    const formData = new FormData();
    formData.append('sample_id', sampleObj.id);
    formData.append('split', sampleObj.split);
    formData.append('threshold', 0.5);

    setLoading(true);
    fetch('/api/predict/stage2', { method: 'POST', body: formData })
      .then((res) => res.json())
      .then((data) => {
        setLoading(false);
        if (data.error) {
          alert(data.error);
          return;
        }
        setSwipeData(data);
        setTimelineStep(0);
      })
      .catch(() => {
        setLoading(false);
        alert('Backend connection error.');
      });
  };

  const timelineFrames = swipeData
    ? [
        {
          label: 'T1 · Earlier',
          description: 'Original satellite image for this paired sample.',
          image: swipeData.t1_image,
        },
        {
          label: 'T2 · Later',
          description: 'Later image of the same paired sample.',
          image: swipeData.t2_image,
        },
        {
          label: 'Predicted forest loss',
          description: 'Red pixels show forest loss predicted between T1 and T2.',
          image: swipeData.deforestation_overlay,
        },
      ]
    : [];
  const selectedTimelineFrame = timelineFrames[timelineStep];

  const s1Iou = analytics?.stage1?.iou ?? FALLBACK_S1_IOU;

  /* Model overlay controls rendered inside the map drawer. */
  const overlayControls = (
    <div className="block">
      <div className="block__head">
        <div className="block__title">
          <ScanLine size={14} color="var(--cyan-bright)" /> Model Overlay
        </div>
        {aiOverlayData && <span className="badge badge--green">Active</span>}
      </div>

      <div className="seg" style={{ width: '100%', marginBottom: 9 }}>
        <button
          type="button"
          className={`seg__btn ${gisMode === 'stage1' ? 'is-active' : ''}`}
          style={{ flex: 1, justifyContent: 'center' }}
          onClick={() => setGisMode('stage1')}
        >
          Stage 1 · Canopy
        </button>
        <button
          type="button"
          className={`seg__btn ${gisMode === 'stage2' ? 'is-active' : ''}`}
          style={{ flex: 1, justifyContent: 'center' }}
          onClick={() => setGisMode('stage2')}
        >
          Stage 2 · Loss
        </button>
      </div>

      <div className="field">
        <label htmlFor="gis-sample">Sample tile</label>
        <select
          id="gis-sample"
          className="select"
          value={selectedGisSample}
          onChange={(e) => setSelectedGisSample(e.target.value)}
        >
          <option value="">— Choose a sample tile —</option>
          {gisMode === 'stage1'
            ? samples.stage1_samples.map((s, idx) => (
                <option key={idx} value={JSON.stringify(s)}>
                  [{s.split}] {s.name}
                </option>
              ))
            : samples.stage2_samples.map((s, idx) => (
                <option key={idx} value={JSON.stringify(s)}>
                  {s.name}
                </option>
              ))}
        </select>
      </div>

      <div className="slider-block" style={{ marginTop: 9 }}>
        <div className="slider-row">
          <span>Threshold</span>
          <span className="slider-value">{gisThreshold.toFixed(2)}</span>
        </div>
        <input
          type="range"
          min="0.1"
          max="0.9"
          step="0.05"
          value={gisThreshold}
          onChange={(e) => setGisThreshold(parseFloat(e.target.value))}
          aria-label="Prediction threshold"
        />
      </div>

      <div className="slider-block">
        <div className="slider-row">
          <span>Mask opacity</span>
          <span className="slider-value">{Math.round(gisOpacity * 100)}%</span>
        </div>
        <input
          type="range"
          min="0.1"
          max="1"
          step="0.05"
          value={gisOpacity}
          onChange={(e) => setGisOpacity(parseFloat(e.target.value))}
          aria-label="Mask opacity"
        />
      </div>

      <button
        type="button"
        className="btn btn--cyan btn--block btn--sm"
        style={{ marginTop: 9 }}
        onClick={handleRunGis}
        disabled={loading}
      >
        {loading ? <span className="btn__spinner" /> : <Play size={13} />}
        Run on sample tile
      </button>

      {aiOverlayData?.metrics && (
        <div className="note note--ok" style={{ marginTop: 8 }}>
          {gisMode === 'stage1' ? (
            <>
              <strong>{aiOverlayData.metrics.forest_hectares} ha</strong> canopy ·{' '}
              {aiOverlayData.metrics.forest_pct}% cover
            </>
          ) : (
            <>
              <strong>{aiOverlayData.metrics.deforestation_hectares} ha</strong> forest loss
            </>
          )}
          <span style={{ color: 'var(--text-faint)' }}>(est. 10 m GSD)</span>
        </div>
      )}
    </div>
  );

  return (
    <div className="app-shell">
      {/* ── Top navigation bar ─────────────────────────────────── */}
      <header className="topbar">
        <div className="topbar__brand">
          <div className="brand-mark">
            <Radar size={20} strokeWidth={2.4} />
          </div>
          <div className="brand-text">
            <span className="brand-title">
              Forest<span>Watch</span> GIS
            </span>
            <span className="brand-sub">Satellite Deforestation Intelligence</span>
          </div>
        </div>

        <div className="topbar__meta">
          <div className={`status-chip ${sentinelStatus === 'configured' ? 'is-live' : sentinelStatus === 'missing' ? 'is-warn' : 'is-off'}`}>
            <span className="status-dot" />
            {sentinelStatus === 'configured' && 'Connected to Sentinel-2 STAC API'}
            {sentinelStatus === 'checking' && 'Checking Sentinel-2 STAC…'}
            {sentinelStatus === 'missing' && 'Sentinel-2 credentials missing'}
            {sentinelStatus === 'unknown' && 'STAC status unknown'}
          </div>

          <div className={`status-chip ${backendStatus === 'online' ? 'is-live' : backendStatus === 'offline' ? 'is-off' : 'is-warn'}`}>
            <span className="status-dot" />
            {backendStatus === 'online' && 'Inference API online'}
            {backendStatus === 'connecting' && 'Connecting to inference API…'}
            {backendStatus === 'offline' && 'Inference API offline'}
          </div>

          <div className="bench-row" style={{ padding: '4px 8px', background: 'transparent', border: '1px dashed var(--border)' }}>
            <span className="bench-badge" title="Stage 1 U-Net validation IoU">
              <Trees size={11} /> Forest U-Net <strong>{s1Iou.toFixed(3)}</strong>
            </span>
            <span className="bench-badge bench-badge--warn" title="Stage 2 Temporal U-Net best validation IoU">
              <Layers size={11} /> Temporal U-Net <strong>{S2_VAL_IOU.toFixed(3)}</strong>
            </span>
          </div>
        </div>

        <div className="mode-switch" role="tablist" aria-label="Application mode">
          <button
            type="button"
            role="tab"
            aria-selected={view === 'map'}
            className={`mode-switch__btn ${view === 'map' ? 'is-active' : ''}`}
            onClick={() => setView('map')}
          >
            <Globe size={14} /> Interactive GIS Map
          </button>
          <button
            type="button"
            role="tab"
            aria-selected={view === 'manual'}
            className={`mode-switch__btn ${view === 'manual' ? 'is-active' : ''}`}
            onClick={() => setView('manual')}
          >
            <Upload size={14} /> Manual Upload (Fallback)
          </button>
        </div>
      </header>

      <main className="app-main">
        {view === 'map' ? (
          <div className="view" style={{ gap: 0 }}>
            <InteractiveMap
              samples={samples}
              aiMode={gisMode}
              aiOpacity={gisOpacity}
              aiOverlayData={aiOverlayData}
              stage2Threshold={s2Threshold}
              onStage2Result={handleLiveStage2Result}
              sidebarTop={overlayControls}
            />
          </div>
        ) : (
          <div className="view">
            <div className="view-head">
              <div>
                <h1 className="view-head__title">Manual Image Upload</h1>
                <p className="view-head__sub">
                  Fallback workbench for offline analysis. Upload a single tile for canopy segmentation, or a
                  co-registered before/after pair for change detection — no live Sentinel-2 access required.
                </p>
              </div>
              <button type="button" className="btn btn--ghost" onClick={() => setView('map')}>
                <MapPin size={13} /> Back to map
              </button>
            </div>

            <div className="manual-tabs" role="tablist" aria-label="Manual workbench">
              {MANUAL_TABS.map(({ id, label, icon: Icon }) => (
                <button
                  key={id}
                  type="button"
                  role="tab"
                  aria-selected={manualTab === id}
                  className={`manual-tab ${manualTab === id ? 'is-active' : ''}`}
                  onClick={() => setManualTab(id)}
                >
                  <Icon size={13} /> {label}
                </button>
              ))}
            </div>

            <div className="manual-scroll">
              {manualTab === 'stage1' && (
                <Stage1View
                  samples={samples}
                  s1Sample={s1Sample}
                  setS1Sample={setS1Sample}
                  s1File={s1File}
                  setS1File={setS1File}
                  s1Threshold={s1Threshold}
                  setS1Threshold={setS1Threshold}
                  s1Results={s1Results}
                  loading={loading}
                  onRun={handleRunStage1}
                />
              )}

              {manualTab === 'stage2' && (
                <Stage2View
                  samples={samples}
                  sentinelStatus={sentinelStatus}
                  s2Sample={s2Sample}
                  setS2Sample={setS2Sample}
                  s2FileT1={s2FileT1}
                  setS2FileT1={setS2FileT1}
                  s2FileT2={s2FileT2}
                  setS2FileT2={setS2FileT2}
                  s2Threshold={s2Threshold}
                  setS2Threshold={setS2Threshold}
                  s2Results={s2Results}
                  loading={loading}
                  onRun={handleRunStage2}
                  onOpenMap={() => {
                    setGisMode('stage2');
                    setView('map');
                  }}
                />
              )}

              {manualTab === 'timeline' && (
                <TimelineView
                  samples={samples}
                  swipeSample={swipeSample}
                  setSwipeSample={setSwipeSample}
                  setSwipeData={setSwipeData}
                  setTimelineStep={setTimelineStep}
                  swipeData={swipeData}
                  timelineFrames={timelineFrames}
                  timelineStep={timelineStep}
                  selectedTimelineFrame={selectedTimelineFrame}
                  loading={loading}
                  onLoad={handleLoadSwipe}
                />
              )}

              {manualTab === 'benchmarks' && <EvaluationPanel analytics={analytics} />}
            </div>
          </div>
        )}
      </main>
    </div>
  );
}

/* ══════════════════════════════════════════════════════════════
   Manual workbench views
   ══════════════════════════════════════════════════════════════ */

function Stage1View({
  samples,
  s1Sample,
  setS1Sample,
  s1File,
  setS1File,
  s1Threshold,
  setS1Threshold,
  s1Results,
  loading,
  onRun,
}) {
  return (
    <>
      <section className="panel">
        <div className="panel__head">
          <div>
            <div className="panel__title">Forest Cover Segmentation</div>
            <div className="block__hint">Single-image binary canopy mask with physical area estimates.</div>
          </div>
          <span className="badge badge--green">
            <ShieldCheck size={11} /> U-Net · 7.76M params
          </span>
        </div>

        <div className="control-grid">
          <div className="field">
            <label htmlFor="s1-sample">Pre-loaded sample</label>
            <select
              id="s1-sample"
              className="select"
              value={s1Sample}
              onChange={(e) => {
                setS1Sample(e.target.value);
                setS1File(null);
              }}
            >
              <option value="">— Choose a sample —</option>
              {samples.stage1_samples.map((s, i) => (
                <option key={i} value={s.path}>
                  [{s.split}] {s.name}
                </option>
              ))}
            </select>
          </div>

          <label className={`upload-zone ${s1File ? 'has-file' : ''}`}>
            <input
              type="file"
              accept="image/*"
              onChange={(e) => {
                if (e.target.files.length > 0) {
                  setS1File(e.target.files[0]);
                  setS1Sample('');
                }
              }}
            />
            <Upload size={17} />
            <span className="upload-zone__label">{s1File ? s1File.name : 'Upload a TIFF / PNG'}</span>
            <span className="upload-zone__hint">Drop-in custom imagery for offline inference</span>
          </label>

          <div className="slider-block">
            <div className="slider-row">
              <span>Confidence threshold</span>
              <span className="slider-value">{s1Threshold.toFixed(2)}</span>
            </div>
            <input
              type="range"
              min="0.1"
              max="0.9"
              step="0.05"
              value={s1Threshold}
              onChange={(e) => setS1Threshold(parseFloat(e.target.value))}
              aria-label="Confidence threshold"
            />
          </div>

          <button type="button" className="btn btn--primary" onClick={onRun} disabled={loading}>
            {loading ? <span className="btn__spinner" /> : <Play size={14} />}
            Run segmentation
          </button>
        </div>
      </section>

      {s1Results && (
        <>
          <div className="stat-grid" style={{ gridTemplateColumns: 'repeat(auto-fit, minmax(190px, 1fr))' }}>
            <div className="stat stat--green">
              <span className="stat__label">
                <Trees size={11} /> Forest canopy
              </span>
              <div className="stat__value">
                {s1Results.forest_hectares}
                <span className="stat__unit">ha</span>
              </div>
              <div className="stat__sub">{s1Results.forest_pct}% of total land cover</div>
            </div>
            <div className="stat stat--neutral">
              <span className="stat__label">Non-forest</span>
              <div className="stat__value">
                {s1Results.non_forest_pct}
                <span className="stat__unit">%</span>
              </div>
              <div className="stat__sub">{s1Results.non_forest_pixels.toLocaleString()} pixels</div>
            </div>
            <div className="stat stat--cyan">
              <span className="stat__label">Total tile</span>
              <div className="stat__value">
                {s1Results.total_hectares}
                <span className="stat__unit">ha</span>
              </div>
              <div className="stat__sub">10 m ground sample distance (est.)</div>
            </div>
          </div>

          <div className="visual-grid">
            <VisualCard title="Original RGB" src={s1Results.original_image} alt="Original satellite image" />
            <VisualCard title="Binary mask" src={s1Results.mask_image} alt="Predicted forest mask" />
            <VisualCard title="Canopy overlay" src={s1Results.overlay_image} alt="Forest overlay" />
          </div>
        </>
      )}
    </>
  );
}

function Stage2View({
  samples,
  sentinelStatus,
  s2Sample,
  setS2Sample,
  s2FileT1,
  setS2FileT1,
  s2FileT2,
  setS2FileT2,
  s2Threshold,
  setS2Threshold,
  s2Results,
  loading,
  onRun,
  onOpenMap,
}) {
  return (
    <>
      <section className="panel">
        <div className="panel__head">
          <div>
            <div className="panel__title">Bi-Temporal Deforestation Detection</div>
            <div className="block__hint">
              9-channel Temporal U-Net consuming T1, T2 and the absolute spectral difference.
            </div>
          </div>
          <span className="badge badge--violet">
            <Layers size={11} /> Temporal U-Net · 9-ch
          </span>
        </div>

        <div className="control-grid">
          <div className="field" style={{ gridColumn: 'span 2' }}>
            <label>Live Sentinel-2 analysis</label>
            <div className="note note--info">
              <Satellite size={13} />
              <span>
                Draw an AOI and pick two acquisition dates on the GIS map. Both images are fetched for the same
                footprint and analysed by the trained Temporal U-Net.
              </span>
            </div>
            {sentinelStatus !== 'configured' && (
              <div className="note note--warn" style={{ marginTop: 6 }}>
                <RefreshCw size={12} />
                <span>
                  {sentinelStatus === 'checking'
                    ? 'Checking Sentinel-2 service configuration…'
                    : sentinelStatus === 'missing'
                      ? 'Live imagery is unavailable until SENTINEL_CLIENT_ID and SENTINEL_CLIENT_SECRET are configured and the backend restarted. Sample pairs and uploads remain available.'
                      : 'Sentinel-2 status could not be checked. Confirm the backend and Copernicus credentials before relying on live imagery.'}
                </span>
              </div>
            )}
            <button type="button" className="btn btn--cyan" style={{ marginTop: 8 }} onClick={onOpenMap}>
              <MapPin size={13} /> Choose AOI and dates on map
            </button>
          </div>

          <div className="field">
            <label htmlFor="s2-sample">Pre-loaded temporal pair</label>
            <select
              id="s2-sample"
              className="select"
              value={s2Sample}
              onChange={(e) => {
                setS2Sample(e.target.value);
                setS2FileT1(null);
                setS2FileT2(null);
              }}
            >
              <option value="">— No sample selected —</option>
              {samples.stage2_samples.map((s, i) => (
                <option key={i} value={JSON.stringify(s)}>
                  {s.name}
                </option>
              ))}
            </select>
          </div>

          <label className={`upload-zone ${s2FileT1 ? 'has-file' : ''}`}>
            <input
              type="file"
              accept="image/*,.tif,.tiff"
              onChange={(e) => {
                if (e.target.files.length > 0) {
                  setS2FileT1(e.target.files[0]);
                  setS2Sample('');
                }
              }}
            />
            <Upload size={16} />
            <span className="upload-zone__label">{s2FileT1 ? `T1 · ${s2FileT1.name}` : 'Upload Before (T1)'}</span>
            <span className="upload-zone__hint">Co-registered with T2</span>
          </label>

          <label className={`upload-zone ${s2FileT2 ? 'has-file' : ''}`}>
            <input
              type="file"
              accept="image/*,.tif,.tiff"
              onChange={(e) => {
                if (e.target.files.length > 0) {
                  setS2FileT2(e.target.files[0]);
                  setS2Sample('');
                }
              }}
            />
            <Upload size={16} />
            <span className="upload-zone__label">{s2FileT2 ? `T2 · ${s2FileT2.name}` : 'Upload After (T2)'}</span>
            <span className="upload-zone__hint">Identical dimensions required</span>
          </label>

          <div className="slider-block">
            <div className="slider-row">
              <span>Deforestation threshold</span>
              <span className="slider-value">{s2Threshold.toFixed(2)}</span>
            </div>
            <input
              type="range"
              min="0.1"
              max="0.9"
              step="0.05"
              value={s2Threshold}
              onChange={(e) => setS2Threshold(parseFloat(e.target.value))}
              aria-label="Deforestation threshold"
            />
          </div>

          <button type="button" className="btn btn--primary" onClick={onRun} disabled={loading}>
            {loading ? <span className="btn__spinner" /> : <Play size={14} />}
            Detect deforestation
          </button>
        </div>
      </section>

      {s2Results && (
        <>
          {s2Results.live_source && (
            <div className="note note--info">
              <Satellite size={13} />
              <span>
                {s2Results.pair_name} · live Sentinel-2 imagery analysed with the Temporal U-Net. Live scenes have
                no ground-truth mask, so accuracy metrics apply to the benchmark split only.
              </span>
            </div>
          )}

          <div className="stat-grid" style={{ gridTemplateColumns: 'repeat(auto-fit, minmax(190px, 1fr))' }}>
            <div className="stat stat--red">
              <span className="stat__label">Forest loss</span>
              <div className="stat__value">
                {s2Results.deforestation_hectares}
                <span className="stat__unit">ha</span>
              </div>
              <div className="stat__sub">{s2Results.deforestation_pct}% tree cover lost</div>
            </div>
            <div className="stat stat--neutral">
              <span className="stat__label">Stable land</span>
              <div className="stat__value">
                {s2Results.stable_pct}
                <span className="stat__unit">%</span>
              </div>
              <div className="stat__sub">{s2Results.stable_pixels.toLocaleString()} pixels</div>
            </div>
            {s2Results.metrics && (
              <div className="stat stat--green">
                <span className="stat__label">
                  <Activity size={11} /> Sample IoU / Dice
                </span>
                <div className="stat__value">
                  {s2Results.metrics.sample_iou} / {s2Results.metrics.sample_dice}
                </div>
                <div className="stat__sub">Ground truth {s2Results.metrics.gt_defor_hectares} ha</div>
              </div>
            )}
          </div>

          <div className="visual-grid" style={{ gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))' }}>
            <VisualCard title="T1 · Before" src={s2Results.t1_image} alt="Before image" />
            <VisualCard title="T2 · After" src={s2Results.t2_image} alt="After image" />
            <VisualCard title="Change mask" src={s2Results.deforestation_mask} alt="Deforestation mask" />
            <VisualCard title="Loss overlay" src={s2Results.deforestation_overlay} alt="Deforestation overlay" />
          </div>

          {s2Results.model_note && (
            <div className="note note--warn">
              <Cpu size={13} />
              <span>{s2Results.model_note}</span>
            </div>
          )}
        </>
      )}
    </>
  );
}

function TimelineView({
  samples,
  swipeSample,
  setSwipeSample,
  setSwipeData,
  setTimelineStep,
  swipeData,
  timelineFrames,
  timelineStep,
  selectedTimelineFrame,
  loading,
  onLoad,
}) {
  return (
    <>
      <section className="panel">
        <div className="panel__head">
          <div>
            <div className="panel__title">Temporal Pair Comparison</div>
            <div className="block__hint">
              Scrub through the earlier image, the later image and the predicted forest-loss overlay.
            </div>
          </div>
        </div>

        <div className="control-grid">
          <div className="field" style={{ gridColumn: 'span 2' }}>
            <label htmlFor="swipe-sample">Temporal image pair</label>
            <select
              id="swipe-sample"
              className="select"
              value={swipeSample}
              onChange={(event) => {
                setSwipeSample(event.target.value);
                setSwipeData(null);
                setTimelineStep(0);
              }}
            >
              <option value="">— Choose a sample pair —</option>
              {samples.stage2_samples.map((sample, index) => (
                <option key={index} value={JSON.stringify(sample)}>
                  {sample.name}
                </option>
              ))}
            </select>
          </div>

          <button type="button" className="btn btn--primary" onClick={onLoad} disabled={loading}>
            {loading ? <span className="btn__spinner" /> : <Play size={14} />}
            Load timeline
          </button>
        </div>
      </section>

      {swipeData ? (
        <section className="panel">
          <div className="panel__head">
            <div>
              <div className="panel__title">{selectedTimelineFrame?.label}</div>
              <div className="block__hint">{selectedTimelineFrame?.description}</div>
            </div>
            <div className="row" style={{ gap: 12 }}>
              <span className="badge badge--dim">{swipeData.pair_name}</span>
              <span className="badge badge--red">
                {swipeData.deforestation_pct}% loss ·{' '}
                {Number(swipeData.deforestation_pixels ?? 0).toLocaleString()} px
              </span>
            </div>
          </div>

          <div className="compare-stage">
            {selectedTimelineFrame && (
              <>
                <span className="compare-badge">{selectedTimelineFrame.label}</span>
                <img
                  src={selectedTimelineFrame.image}
                  alt={`${selectedTimelineFrame.label} for ${swipeData.pair_name || 'selected pair'}`}
                />
              </>
            )}
          </div>

          <div className="slider-block" style={{ marginTop: 12 }}>
            <input
              type="range"
              min="0"
              max={Math.max(timelineFrames.length - 1, 0)}
              step="1"
              value={timelineStep}
              aria-label="Timeline timepoint"
              aria-valuetext={selectedTimelineFrame?.label || ''}
              onChange={(event) => setTimelineStep(Number(event.target.value))}
            />
          </div>

          <div className="timeline-ticks" style={{ marginTop: 8 }}>
            {timelineFrames.map((frame, index) => (
              <button
                key={frame.label}
                type="button"
                className={`timeline-tick ${timelineStep === index ? 'is-active' : ''}`}
                onClick={() => setTimelineStep(index)}
              >
                {frame.label}
              </button>
            ))}
          </div>
        </section>
      ) : (
        <div className="empty-state">
          <GitCompare size={22} />
          Select a temporal pair and load the timeline to compare T1, T2 and the predicted change.
        </div>
      )}
    </>
  );
}

function VisualCard({ title, src, alt }) {
  if (!src) return null;
  return (
    <div className="visual-card">
      <div className="visual-card__head">{title}</div>
      <img className="visual-card__img" src={src} alt={alt} />
    </div>
  );
}
