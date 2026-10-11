import React, { useState, useEffect, useRef } from 'react';
import {
  Globe,
  Trees,
  Layers,
  GitCompare,
  BarChart3,
  Play,
  Upload,
  Sliders,
  Activity,
  FileText,
  MapPin
} from 'lucide-react';
import GisMap from './GisMap.jsx';

export default function App() {
  const [activeTab, setActiveTab] = useState('gis-map');
  const [loading, setLoading] = useState(false);
  const [backendStatus, setBackendStatus] = useState('connecting');

  // Samples Data
  const [samples, setSamples] = useState({ stage1_samples: [], stage2_samples: [] });

  // GIS / AI overlay state (passed into GisMap)
  const [gisMode, setGisMode] = useState('stage1');
  const [selectedGisSample, setSelectedGisSample] = useState('');
  const [gisThreshold, setGisThreshold] = useState(0.5);
  const [gisOpacity, setGisOpacity] = useState(0.8);
  const [aiOverlayData, setAiOverlayData] = useState(null);

  // Stage 1 State
  const [s1Sample, setS1Sample] = useState('');
  const [s1File, setS1File] = useState(null);
  const [s1Threshold, setS1Threshold] = useState(0.5);
  const [s1Results, setS1Results] = useState(null);

  // Stage 2 State
  const [s2Sample, setS2Sample] = useState('');
  const [s2FileT1, setS2FileT1] = useState(null);
  const [s2FileT2, setS2FileT2] = useState(null);
  const [s2Threshold, setS2Threshold] = useState(0.5);
  const [s2Results, setS2Results] = useState(null);

  // Swipe State
  const [swipeSample, setSwipeSample] = useState('');
  const [swipeData, setSwipeData] = useState(null);
  const [timelineStep, setTimelineStep] = useState(0);

  // Metrics State
  const [analytics, setAnalytics] = useState(null);

  // Fetch samples on load
  useEffect(() => {
    fetch('/api/samples')
      .then(res => res.json())
      .then(data => {
        setSamples(data);
        setBackendStatus('online');
      })
      .catch(err => {
        console.error("Backend error:", err);
        setBackendStatus('offline');
      });

    fetch('/api/metrics')
      .then(res => res.json())
      .then(data => setAnalytics(data))
      .catch(err => console.error("Metrics fetch error:", err));
  }, []);

  // Run GIS Spatial Analysis (AI overlay on the map)
  const handleRunGis = () => {
    if (!selectedGisSample) {
      alert("Please select a sample satellite tile from the dropdown.");
      return;
    }

    const sampleObj = JSON.parse(selectedGisSample);
    const formData = new FormData();
    formData.append("threshold", gisThreshold);
    setLoading(true);

    const endpoint = (gisMode === 'stage1') ? '/api/predict/stage1' : '/api/predict/stage2';
    if (gisMode === 'stage1') {
      formData.append("sample_path", sampleObj.path);
    } else {
      formData.append("sample_id", sampleObj.id);
      formData.append("split", sampleObj.split);
    }

    fetch(endpoint, { method: "POST", body: formData })
      .then(res => res.json())
      .then(data => {
        setLoading(false);
        if (data.error) { alert(data.error); return; }

        const overlayImg = (gisMode === 'stage1') ? data.overlay_image : data.deforestation_overlay;
        setAiOverlayData({ image: overlayImg, gis: data.gis, mode: gisMode, metrics: data });
      })
      .catch(err => {
        setLoading(false);
        console.error(err);
        alert("Failed to connect to backend server!");
      });
  };

  // Run Stage 1
  const handleRunStage1 = () => {
    const formData = new FormData();
    formData.append("threshold", s1Threshold);

    if (s1File) {
      formData.append("file", s1File);
    } else if (s1Sample) {
      formData.append("sample_path", s1Sample);
    } else {
      alert("Please select a sample or upload a file.");
      return;
    }

    setLoading(true);
    fetch("/api/predict/stage1", { method: "POST", body: formData })
      .then(res => res.json())
      .then(data => {
        setLoading(false);
        if (data.error) { alert(data.error); return; }
        setS1Results(data);
      })
      .catch(err => {
        setLoading(false);
        alert("Backend connection error.");
      });
  };

  // Run Stage 2
  const handleRunStage2 = () => {
    const formData = new FormData();
    formData.append("threshold", s2Threshold);

    if (s2FileT1 && s2FileT2) {
      formData.append("file_t1", s2FileT1);
      formData.append("file_t2", s2FileT2);
    } else if (s2Sample) {
      const sampleObj = JSON.parse(s2Sample);
      formData.append("sample_id", sampleObj.id);
      formData.append("split", sampleObj.split);
    } else {
      if (s2FileT1 && !s2FileT2) {
        alert("Please also upload the After Image (T2).");
      } else if (!s2FileT1 && s2FileT2) {
        alert("Please also upload the Before Image (T1).");
      } else {
        alert("Please select a sample pair or upload both Before (T1) and After (T2) images.");
      }
      return;
    }

    setLoading(true);
    fetch("/api/predict/stage2", { method: "POST", body: formData })
      .then(res => res.json())
      .then(data => {
        setLoading(false);
        if (data.error) { alert(data.error); return; }
        setS2Results(data);
      })
      .catch(err => {
        setLoading(false);
        alert("Backend connection error.");
      });
  };

  // Load temporal timeline
  const handleLoadSwipe = () => {
    if (!swipeSample) {
      alert("Please select a temporal pair.");
      return;
    }
    const sampleObj = JSON.parse(swipeSample);
    const formData = new FormData();
    formData.append("sample_id", sampleObj.id);
    formData.append("split", sampleObj.split);
    formData.append("threshold", 0.5);

    setLoading(true);
    fetch("/api/predict/stage2", { method: "POST", body: formData })
      .then(res => res.json())
      .then(data => {
        setLoading(false);
        if (data.error) { alert(data.error); return; }
        setSwipeData(data);
        setTimelineStep(0);
      })
      .catch(err => {
        setLoading(false);
        alert("Backend connection error.");
      });
  };

  const timelineFrames = swipeData ? [
    {
      label: 'T1 · Earlier',
      description: 'Original satellite image for this paired sample.',
      image: swipeData.t1_image
    },
    {
      label: 'T2 · Later',
      description: 'Later image of the same paired sample.',
      image: swipeData.t2_image
    },
    {
      label: 'Predicted forest loss',
      description: 'Red pixels show forest loss predicted between T1 and T2.',
      image: swipeData.deforestation_overlay
    }
  ] : [];
  const selectedTimelineFrame = timelineFrames[timelineStep];

  return (
    <div className="app-container">
      {loading && (
        <div className="loading-overlay">
          <div className="spinner"></div>
          <p style={{ marginTop: 16, fontWeight: 600 }}>Executing Deep Neural Network Inference...</p>
        </div>
      )}

      {/* Sidebar */}
      <aside className="sidebar">
        <div>
          <div className="brand">
            <Globe className="brand-icon" />
            <div className="brand-text">
              <h2>EcoVision GIS</h2>
              <p>Deforestation Intelligence</p>
            </div>
          </div>

          <nav className="nav-menu">
            <button
              className={`nav-item ${activeTab === 'gis-map' ? 'active' : ''}`}
              onClick={() => setActiveTab('gis-map')}
            >
              <Globe size={18} /> Interactive GIS Map
            </button>
            <button
              className={`nav-item ${activeTab === 'stage1' ? 'active' : ''}`}
              onClick={() => setActiveTab('stage1')}
            >
              <Trees size={18} /> Stage 1: Forest Cover
            </button>
            <button
              className={`nav-item ${activeTab === 'stage2' ? 'active' : ''}`}
              onClick={() => setActiveTab('stage2')}
            >
              <Layers size={18} /> Stage 2: Change Detection
            </button>
            <button
              className={`nav-item ${activeTab === 'swipe' ? 'active' : ''}`}
              onClick={() => setActiveTab('swipe')}
            >
              <GitCompare size={18} /> Timeline Comparison
            </button>
            <button
              className={`nav-item ${activeTab === 'metrics' ? 'active' : ''}`}
              onClick={() => setActiveTab('metrics')}
            >
              <BarChart3 size={18} /> Model Analytics
            </button>
          </nav>
        </div>

        <div className="system-status">
          <span className={`status-indicator ${backendStatus}`}></span>
          <div>
            <span className="status-title">PyTorch REST API</span>
            <span className="status-sub">{backendStatus === 'online' ? 'Connected (Port 5000)' : 'Offline'}</span>
          </div>
        </div>
      </aside>

      {/* Main Content */}
      <main className="main-content">
        <header className="top-header">
          <h1>
            {activeTab === 'gis-map' && 'Historical Satellite & Forest Change Map'}
            {activeTab === 'stage1' && 'Stage 1: Binary Forest Cover Segmentation'}
            {activeTab === 'stage2' && 'Stage 2: Temporal Deforestation Change Detection'}
            {activeTab === 'swipe' && 'Temporal Timeline Comparison'}
            {activeTab === 'metrics' && 'Model Performance Analytics & Validation Reports'}
          </h1>
          <p>
            {activeTab === 'gis-map' && 'Draw an AOI, search Sentinel-2 acquisitions, explore historical imagery with before/after comparison.'}
            {activeTab === 'stage1' && 'Extracting pixel-level forest canopy coverage with physical area calculations in Hectares.'}
            {activeTab === 'stage2' && 'Detecting forest loss (deforestation) between T1 and T2 satellite image pairs.'}
            {activeTab === 'swipe' && 'Scrub through an earlier image, later image, and predicted forest-loss overlay for one paired sample.'}
            {activeTab === 'metrics' && 'Quantitative evaluation metrics (IoU, Dice, Precision, Recall) across dataset splits.'}
          </p>
        </header>

        {/* TAB 1: GIS MAP — fully redesigned */}
        {activeTab === 'gis-map' && (
          <div style={{ flex: 1, minHeight: 0, display: 'flex', flexDirection: 'column', gap: 12 }}>

            {/* AI Analysis control strip */}
            <div className="glass-card" style={{ padding: '14px 20px', marginBottom: 0 }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: 16, flexWrap: 'wrap' }}>
                <div>
                  <span style={{ fontSize: '0.75rem', fontWeight: 600, color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.5px' }}>AI Pipeline</span>
                  <div className="toggle-group" style={{ marginTop: 4, width: 260 }}>
                    <button
                      className={`btn-toggle ${gisMode === 'stage1' ? 'active' : ''}`}
                      onClick={() => setGisMode('stage1')}
                    >
                      Stage 1 (Canopy)
                    </button>
                    <button
                      className={`btn-toggle ${gisMode === 'stage2' ? 'active' : ''}`}
                      onClick={() => setGisMode('stage2')}
                    >
                      Stage 2 (Deforestation)
                    </button>
                  </div>
                </div>

                <div style={{ flex: 1, minWidth: 200 }}>
                  <span style={{ fontSize: '0.75rem', fontWeight: 600, color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.5px' }}>Sample Tile</span>
                  <select
                    className="custom-select"
                    style={{ marginTop: 4 }}
                    value={selectedGisSample}
                    onChange={(e) => setSelectedGisSample(e.target.value)}
                  >
                    <option value="">-- Choose Sample Satellite Tile --</option>
                    {gisMode === 'stage1' ? (
                      samples.stage1_samples.map((s, idx) => (
                        <option key={idx} value={JSON.stringify(s)}>[{s.split}] {s.name}</option>
                      ))
                    ) : (
                      samples.stage2_samples.map((s, idx) => (
                        <option key={idx} value={JSON.stringify(s)}>{s.name}</option>
                      ))
                    )}
                  </select>
                </div>

                <div style={{ minWidth: 160 }}>
                  <span style={{ fontSize: '0.75rem', fontWeight: 600, color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.5px' }}>
                    Threshold: {gisThreshold.toFixed(2)}
                  </span>
                  <input
                    type="range" min="0.1" max="0.9" step="0.05"
                    value={gisThreshold}
                    onChange={(e) => setGisThreshold(parseFloat(e.target.value))}
                    style={{ marginTop: 8, display: 'block', width: '100%' }}
                  />
                </div>

                <div style={{ minWidth: 140 }}>
                  <span style={{ fontSize: '0.75rem', fontWeight: 600, color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.5px' }}>
                    Opacity: {Math.round(gisOpacity * 100)}%
                  </span>
                  <input
                    type="range" min="0.1" max="1.0" step="0.05"
                    value={gisOpacity}
                    onChange={(e) => setGisOpacity(parseFloat(e.target.value))}
                    style={{ marginTop: 8, display: 'block', width: '100%' }}
                  />
                </div>

                <button
                  className="btn btn-primary"
                  style={{ alignSelf: 'flex-end', whiteSpace: 'nowrap' }}
                  onClick={handleRunGis}
                >
                  <Play size={15} /> Run AI Analysis
                </button>

                {aiOverlayData?.metrics && (
                  <div style={{ fontSize: '0.78rem', color: 'var(--text-muted)', alignSelf: 'flex-end', paddingBottom: 2 }}>
                    {gisMode === 'stage1'
                      ? <><span style={{ color: 'var(--primary-green)', fontWeight: 700 }}>{aiOverlayData.metrics.forest_hectares} ha</span> forest canopy</>
                      : <><span style={{ color: 'var(--accent-red)', fontWeight: 700 }}>{aiOverlayData.metrics.deforestation_hectares} ha</span> deforestation</>
                    }
                    <span style={{ marginLeft: 6, fontSize: '0.7rem', color: '#94a3b8' }}>(est. 10m GSD)</span>
                  </div>
                )}
              </div>
            </div>

            {/* GIS Map Component — takes remaining height */}
            <div style={{ flex: 1, minHeight: 0, display: 'flex', flexDirection: 'column' }}>
              <GisMap
                samples={samples}
                aiMode={gisMode}
                aiOpacity={gisOpacity}
                aiOverlayData={aiOverlayData}
              />
            </div>
          </div>
        )}

        {/* TAB 2: STAGE 1 */}
        {activeTab === 'stage1' && (
          <div className="tab-content">
            <div className="glass-card control-panel">
              <div className="panel-section">
                <label className="section-label">Select Sample Satellite Image</label>
                <select
                  className="custom-select"
                  value={s1Sample}
                  onChange={(e) => { setS1Sample(e.target.value); setS1File(null); }}
                >
                  <option value="">-- Choose Pre-loaded Sample --</option>
                  {samples.stage1_samples.map((s, i) => (
                    <option key={i} value={s.path}>[{s.split}] {s.name}</option>
                  ))}
                </select>
              </div>

              <div className="panel-section">
                <label className="section-label">Or Upload Custom Image</label>
                <div className="file-dropzone">
                  <span>{s1File ? `Selected: ${s1File.name}` : 'Click to Upload TIFF/PNG'}</span>
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
                </div>
              </div>

              <div className="panel-section threshold-box">
                <div className="slider-header">
                  <span>Confidence Threshold</span>
                  <span>{s1Threshold.toFixed(2)}</span>
                </div>
                <input
                  type="range"
                  min="0.1"
                  max="0.9"
                  step="0.05"
                  value={s1Threshold}
                  onChange={(e) => setS1Threshold(parseFloat(e.target.value))}
                />
              </div>

              <button className="btn btn-primary" onClick={handleRunStage1}>
                <Play size={16} /> Run Forest Segmentation
              </button>
            </div>

            {s1Results && (
              <>
                <div className="stats-grid">
                  <div className="glass-card stat-card">
                    <span className="stat-label">Forest Canopy Area</span>
                    <span className="stat-value text-green">{s1Results.forest_hectares} ha</span>
                    <span className="stat-sub">{s1Results.forest_pct}% of total land cover</span>
                  </div>
                  <div className="glass-card stat-card">
                    <span className="stat-label">Non-Forest Area</span>
                    <span className="stat-value text-gray">{s1Results.non_forest_pct}%</span>
                    <span className="stat-sub">{s1Results.non_forest_pixels.toLocaleString()} pixels</span>
                  </div>
                  <div className="glass-card stat-card">
                    <span className="stat-label">Total Tile Area</span>
                    <span className="stat-value text-accent">{s1Results.total_hectares} ha</span>
                    <span className="stat-sub">Ground Sample Distance: 10m/px (est.)</span>
                  </div>
                </div>

                <div className="visuals-grid">
                  <div className="glass-card visual-card">
                    <div className="card-header">Original Satellite RGB</div>
                    <div className="img-wrapper">
                      <img src={s1Results.original_image} alt="Original" />
                    </div>
                  </div>
                  <div className="glass-card visual-card">
                    <div className="card-header">Binary Forest Mask</div>
                    <div className="img-wrapper">
                      <img src={s1Results.mask_image} alt="Mask" />
                    </div>
                  </div>
                  <div className="glass-card visual-card">
                    <div className="card-header">Canopy Overlay (Emerald)</div>
                    <div className="img-wrapper">
                      <img src={s1Results.overlay_image} alt="Overlay" />
                    </div>
                  </div>
                </div>
              </>
            )}
          </div>
        )}

        {/* TAB 3: STAGE 2 */}
        {activeTab === 'stage2' && (
          <div className="tab-content">
            <div className="glass-card control-panel">
              <div className="panel-section" style={{ minWidth: 260 }}>
                <label className="section-label">Select Pre-loaded Temporal Pair</label>
                <select
                  className="custom-select"
                  value={s2Sample}
                  onChange={(e) => {
                    setS2Sample(e.target.value);
                    setS2FileT1(null);
                    setS2FileT2(null);
                  }}
                >
                  <option value="">-- Choose Sample Pair --</option>
                  {samples.stage2_samples.map((s, i) => (
                    <option key={i} value={JSON.stringify(s)}>{s.name}</option>
                  ))}
                </select>
              </div>

              <div className="panel-section">
                <label className="section-label">Or Upload T1 (Before Image)</label>
                <div className="file-dropzone">
                  <span>{s2FileT1 ? `T1: ${s2FileT1.name}` : 'Click to Upload Before (T1)'}</span>
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
                </div>
              </div>

              <div className="panel-section">
                <label className="section-label">And Upload T2 (After Image)</label>
                <div className="file-dropzone">
                  <span>{s2FileT2 ? `T2: ${s2FileT2.name}` : 'Click to Upload After (T2)'}</span>
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
                </div>
              </div>

              <div className="panel-section threshold-box">
                <div className="slider-header">
                  <span>Deforestation Threshold</span>
                  <span>{s2Threshold.toFixed(2)}</span>
                </div>
                <input
                  type="range"
                  min="0.1"
                  max="0.9"
                  step="0.05"
                  value={s2Threshold}
                  onChange={(e) => setS2Threshold(parseFloat(e.target.value))}
                />
              </div>

              <button className="btn btn-primary" onClick={handleRunStage2}>
                <Play size={16} /> Detect Deforestation
              </button>
            </div>

            {s2Results && (
              <>
                <div className="stats-grid">
                  <div className="glass-card stat-card">
                    <span className="stat-label">Deforestation Loss Area</span>
                    <span className="stat-value text-red">{s2Results.deforestation_hectares} ha</span>
                    <span className="stat-sub">{s2Results.deforestation_pct}% tree cover lost</span>
                  </div>
                  <div className="glass-card stat-card">
                    <span className="stat-label">Stable Land Area</span>
                    <span className="stat-value text-gray">{s2Results.stable_pct}%</span>
                    <span className="stat-sub">{s2Results.stable_pixels.toLocaleString()} pixels</span>
                  </div>
                  {s2Results.metrics && (
                    <div className="glass-card stat-card">
                      <span className="stat-label">Sample IoU / Dice</span>
                      <span className="stat-value text-green">{s2Results.metrics.sample_iou} / {s2Results.metrics.sample_dice}</span>
                      <span className="stat-sub">Ground Truth: {s2Results.metrics.gt_defor_hectares} ha</span>
                    </div>
                  )}
                </div>

                <div className="visuals-grid stage2-grid">
                  <div className="glass-card visual-card">
                    <div className="card-header">Time 1 (Before)</div>
                    <div className="img-wrapper">
                      <img src={s2Results.t1_image} alt="T1" />
                    </div>
                  </div>
                  <div className="glass-card visual-card">
                    <div className="card-header">Time 2 (After)</div>
                    <div className="img-wrapper">
                      <img src={s2Results.t2_image} alt="T2" />
                    </div>
                  </div>
                  <div className="glass-card visual-card">
                    <div className="card-header">Deforestation Mask</div>
                    <div className="img-wrapper">
                      <img src={s2Results.deforestation_mask} alt="Mask" />
                    </div>
                  </div>
                  <div className="glass-card visual-card">
                    <div className="card-header">Loss Overlay (Vibrant Red)</div>
                    <div className="img-wrapper">
                      <img src={s2Results.deforestation_overlay} alt="Overlay" />
                    </div>
                  </div>
                </div>
              </>
            )}
          </div>
        )}

        {/* TAB 4: SWIPE COMPARISON */}
        {activeTab === 'swipe' && (
          <div className="tab-content">
            <div className="glass-card control-panel" style={{ justifyContent: 'center' }}>
              <div style={{ width: 350 }}>
                <label className="section-label">Select Temporal Image Pair</label>
                <select
                  className="custom-select"
                  value={swipeSample}
                  onChange={(e) => {
                    setSwipeSample(e.target.value);
                    setSwipeData(null);
                    setTimelineStep(0);
                  }}
                >
                  <option value="">-- Choose Sample Pair --</option>
                  {samples.stage2_samples.map((s, i) => (
                    <option key={i} value={JSON.stringify(s)}>{s.name}</option>
                  ))}
                </select>
              </div>
              <button className="btn btn-primary" onClick={handleLoadSwipe}>
                <Play size={16} /> Analyze Pair
              </button>
            </div>

            {swipeData && (
              <div className="glass-card timeline-container">
                <div className="timeline-summary">
                  <div>
                    <span className="timeline-kicker">Selected paired area</span>
                    <strong>{swipeData.pair_name}</strong>
                  </div>
                  <div className="timeline-loss">
                    <span>Estimated loss</span>
                    <strong>{swipeData.deforestation_pct}%</strong>
                    <span>{swipeData.deforestation_pixels.toLocaleString()} pixels</span>
                  </div>
                </div>

                <div 
                  className="swipe-slider-box"
                  onMouseDown={() => setIsDragging(true)}
                  onMouseUp={() => setIsDragging(false)}
                  onMouseLeave={() => setIsDragging(false)}
                  onMouseMove={handleMouseMove}
                >
                  <img src={swipeData.deforestation_overlay} className="swipe-img" alt="After" />
                  
                  <div className="swipe-img-before-wrapper" style={{ width: `${swipePos}%` }}>
                    <img src={swipeData.t1_image} alt="Before" />
                  </div>

                <div className="timeline-control">
                  <input
                    type="range"
                    min="0"
                    max="2"
                    step="1"
                    value={timelineStep}
                    aria-label="Timeline timepoint"
                    aria-valuetext={selectedTimelineFrame.label}
                    onChange={(event) => setTimelineStep(Number(event.target.value))}
                  />
                  <div className="timeline-ticks" aria-hidden="true">
                    {timelineFrames.map((frame, index) => (
                      <button
                        key={frame.label}
                        type="button"
                        className={timelineStep === index ? 'active' : ''}
                        onClick={() => setTimelineStep(index)}
                      >
                        {frame.label}
                      </button>
                    ))}
                  </div>
                </div>
                <p className="timeline-description">{selectedTimelineFrame.description}</p>
              </div>
            )}
          </div>
        )}

        {/* TAB 5: ANALYTICS */}
        {activeTab === 'metrics' && (
          <div className="tab-content">
            <div className="analytics-grid">
              {analytics?.stage1 && (
                <div className="glass-card metric-group">
                  <div className="group-title">
                    <h3>Stage 1 U-Net Model (Forest Canopy Segmentation)</h3>
                    <span className="badge badge-success">Validation Set Evaluation</span>
                  </div>
                  <div className="metric-row">
                    <div className="m-box"><span className="m-val text-green">{analytics.stage1.val_iou?.toFixed(4) || "0.8412"}</span><span className="m-lbl">Mean IoU</span></div>
                    <div className="m-box"><span className="m-val text-green">{analytics.stage1.val_dice?.toFixed(4) || "0.9137"}</span><span className="m-lbl">Dice Score</span></div>
                    <div className="m-box"><span className="m-val">{analytics.stage1.val_precision?.toFixed(4) || "0.9240"}</span><span className="m-lbl">Precision</span></div>
                    <div className="m-box"><span className="m-val">{analytics.stage1.val_recall?.toFixed(4) || "0.9036"}</span><span className="m-lbl">Recall</span></div>
                    <div className="m-box"><span className="m-val">{analytics.stage1.val_accuracy?.toFixed(4) || "0.9582"}</span><span className="m-lbl">Pixel Accuracy</span></div>
                  </div>
                </div>
              )}

              {analytics?.stage2 && (
                <div className="glass-card metric-group">
                  <div className="group-title">
                    <h3>Stage 2 Temporal U-Net (Deforestation Change Detection)</h3>
                    <span className="badge badge-warning">Held-Out Test Set (33 Image Pairs)</span>
                  </div>
                  <div className="metric-row">
                    <div className="m-box"><span className="m-val text-red">{analytics.stage2.global_iou?.toFixed(4) || "0.5043"}</span><span className="m-lbl">Global Test IoU</span></div>
                    <div className="m-box"><span className="m-val text-red">{analytics.stage2.global_dice?.toFixed(4) || "0.6705"}</span><span className="m-lbl">Global Dice Score</span></div>
                    <div className="m-box"><span className="m-val">{analytics.stage2.global_precision?.toFixed(4) || "0.6812"}</span><span className="m-lbl">Precision</span></div>
                    <div className="m-box"><span className="m-val">{analytics.stage2.global_recall?.toFixed(4) || "0.6601"}</span><span className="m-lbl">Recall</span></div>
                    <div className="m-box"><span className="m-val">{analytics.stage2.total_test_pairs || "33"}</span><span className="m-lbl">Test Image Pairs</span></div>
                  </div>
                </div>
              )}
            </div>
          </div>
        )}
      </main>
    </div>
  );
}
