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
import L from 'leaflet';

export default function App() {
  const [activeTab, setActiveTab] = useState('gis-map');
  const [loading, setLoading] = useState(false);
  const [backendStatus, setBackendStatus] = useState('connecting');

  // Samples Data
  const [samples, setSamples] = useState({ stage1_samples: [], stage2_samples: [] });
  const [gisMode, setGisMode] = useState('stage1');
  const [selectedGisSample, setSelectedGisSample] = useState('');
  const [gisThreshold, setGisThreshold] = useState(0.5);
  const [gisOpacity, setGisOpacity] = useState(0.8);
  const [gisMetrics, setGisMetrics] = useState(null);

  // Sentinel-2 Preview State
  const [sentinelLoading, setSentinelLoading] = useState(false);
  const [sentinelStatusMsg, setSentinelStatusMsg] = useState('');
  const [sentinelAcquisition, setSentinelAcquisition] = useState(null);
  const [sentinelAcquisitions, setSentinelAcquisitions] = useState([]);
  const [selectedAcquisitionId, setSelectedAcquisitionId] = useState(null);

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
  const [swipePos, setSwipePos] = useState(50);
  const [isDragging, setIsDragging] = useState(false);

  // Metrics State
  const [analytics, setAnalytics] = useState(null);

  // Map refs
  const mapRef = useRef(null);
  const mapInstanceRef = useRef(null);
  const overlayRef = useRef(null);
  const boundsRef = useRef(null);

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

  // Initialize Leaflet Map
  useEffect(() => {
    if (activeTab === 'gis-map' && mapRef.current && !mapInstanceRef.current) {
      const esriTile = L.tileLayer("https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}", {
        maxZoom: 18,
        attribution: "Esri Satellite"
      });

      const osmTile = L.tileLayer("https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png", {
        maxZoom: 19,
        attribution: "OpenStreetMap"
      });

      const map = L.map(mapRef.current, {
        center: [-3.4653, -62.2159],
        zoom: 13,
        layers: [esriTile]
      });

      L.control.layers({
        "Satellite (Esri)": esriTile,
        "Street Map (OSM)": osmTile
      }).addTo(map);

      mapInstanceRef.current = map;
    }

    if (activeTab === 'gis-map' && mapInstanceRef.current) {
      setTimeout(() => mapInstanceRef.current.invalidateSize(), 150);
    }
  }, [activeTab]);

  // Update Opacity live
  useEffect(() => {
    if (overlayRef.current) {
      overlayRef.current.setOpacity(gisOpacity);
    }
  }, [gisOpacity]);

  // Run GIS Spatial Analysis
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

        setGisMetrics(data);

        // Render Leaflet Overlay
        const map = mapInstanceRef.current;
        if (map && data.gis) {
          if (overlayRef.current) map.removeLayer(overlayRef.current);
          if (boundsRef.current) map.removeLayer(boundsRef.current);

          const bounds = L.latLngBounds(data.gis.bounds);

          boundsRef.current = L.rectangle(bounds, {
            color: "#10b981",
            weight: 2,
            fill: false,
            dashArray: "4, 4"
          }).addTo(map);

          const overlayImg = (gisMode === 'stage1') ? data.overlay_image : data.deforestation_overlay;
          overlayRef.current = L.imageOverlay(overlayImg, bounds, { opacity: gisOpacity }).addTo(map);

          map.fitBounds(bounds, { padding: [30, 30] });
        }
      })
      .catch(err => {
        setLoading(false);
        console.error(err);
        alert("Failed to connect to backend server!");
      });
  };

  // Load specific Sentinel-2 acquisition onto Leaflet map
  const loadSentinelAcquisition = (acq) => {
    setSelectedAcquisitionId(acq.id);
    setSentinelAcquisition(acq);
    setSentinelLoading(true);
    setSentinelStatusMsg("Loading Sentinel image...");

    const lat = -3.4653;
    const lon = -62.2159;

    fetch(`/api/sentinel/preview?acquisition_id=${encodeURIComponent(acq.id)}&lat=${lat}&lon=${lon}&format=json`)
      .then(res => res.json().then(pData => ({ ok: res.ok, pData })))
      .then(({ ok, pData }) => {
        setSentinelLoading(false);

        if (!ok || !pData.success || !pData.image) {
          setSentinelStatusMsg(pData.error || "Failed to load Sentinel preview image.");
          return;
        }

        setSentinelStatusMsg("Sentinel-2 imagery loaded");

        // Display on Leaflet map using existing image overlay mechanism
        const map = mapInstanceRef.current;
        if (map) {
          if (overlayRef.current) map.removeLayer(overlayRef.current);
          if (boundsRef.current) map.removeLayer(boundsRef.current);

          const bbox = pData.bbox || [-62.2274, -3.4768, -62.2044, -3.4538];
          const bounds = L.latLngBounds([
            [bbox[1], bbox[0]],
            [bbox[3], bbox[2]]
          ]);

          boundsRef.current = L.rectangle(bounds, {
            color: "#3b82f6",
            weight: 2,
            fill: false,
            dashArray: "4, 4"
          }).addTo(map);

          overlayRef.current = L.imageOverlay(pData.image, bounds, { opacity: gisOpacity }).addTo(map);
          map.fitBounds(bounds, { padding: [30, 30] });
        }
      })
      .catch(err => {
        setSentinelLoading(false);
        setSentinelStatusMsg(err.message || "Failed to connect to Sentinel API.");
      });
  };

  // Load Sentinel-2 Live Satellite Preview & fetch historical acquisitions
  const handleLoadSentinelPreview = () => {
    const lat = -3.4653;
    const lon = -62.2159;
    const startDate = '2024-08-01';
    const endDate = '2024-08-31';

    setSentinelLoading(true);
    setSentinelStatusMsg("Searching Sentinel imagery...");
    setSentinelAcquisition(null);
    setSentinelAcquisitions([]);
    setSelectedAcquisitionId(null);

    fetch(`/api/sentinel/search?lat=${lat}&lon=${lon}&start_date=${startDate}&end_date=${endDate}`)
      .then(res => res.json().then(data => ({ ok: res.ok, data })))
      .then(({ ok, data }) => {
        if (!ok || !data.success || !data.acquisitions || data.acquisitions.length === 0) {
          setSentinelLoading(false);
          setSentinelStatusMsg(data.message || data.error || "No matching Sentinel-2 acquisitions found.");
          return;
        }

        // Sort acquisitions chronologically (oldest to newest) for timeline display
        const sorted = [...data.acquisitions].sort((a, b) => (a.datetime || '').localeCompare(b.datetime || ''));
        setSentinelAcquisitions(sorted);

        // Automatically load the first acquisition into view
        loadSentinelAcquisition(sorted[0]);
      })
      .catch(err => {
        setSentinelLoading(false);
        setSentinelStatusMsg(err.message || "Failed to connect to Sentinel API.");
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

  // Load Swipe
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
      })
      .catch(err => {
        setLoading(false);
        alert("Backend connection error.");
      });
  };

  const handleMouseMove = (e) => {
    if (!isDragging) return;
    const box = e.currentTarget.getBoundingClientRect();
    let posX = e.clientX - box.left;
    if (posX < 0) posX = 0;
    if (posX > box.width) posX = box.width;
    setSwipePos((posX / box.width) * 100);
  };

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
              <GitCompare size={18} /> Swipe Comparison
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
            {activeTab === 'gis-map' && 'Interactive GIS Satellite Map Viewer'}
            {activeTab === 'stage1' && 'Stage 1: Binary Forest Cover Segmentation'}
            {activeTab === 'stage2' && 'Stage 2: Temporal Deforestation Change Detection'}
            {activeTab === 'swipe' && 'Temporal Swipe Comparison Viewer'}
            {activeTab === 'metrics' && 'Model Performance Analytics & Validation Reports'}
          </h1>
          <p>
            {activeTab === 'gis-map' && 'Geospatial analysis of Amazon basin canopy cover & loss using Leaflet.js and PyTorch U-Net.'}
            {activeTab === 'stage1' && 'Extracting pixel-level forest canopy coverage with physical area calculations in Hectares.'}
            {activeTab === 'stage2' && 'Detecting forest loss (deforestation) between T1 and T2 satellite image pairs.'}
            {activeTab === 'swipe' && 'Interactive split-screen swipe tool comparing Before (T1) vs. After (T2 + Deforestation Overlay).'}
            {activeTab === 'metrics' && 'Quantitative evaluation metrics (IoU, Dice, Precision, Recall) across dataset splits.'}
          </p>
        </header>

        {/* TAB 1: GIS MAP */}
        {activeTab === 'gis-map' && (
          <div className="tab-content">
            <div className="gis-layout">
              <div className="gis-sidebar">
                <div className="glass-card">
                  <h3 className="panel-heading">GIS Control Panel</h3>
                  
                  <div className="panel-section" style={{ marginTop: 12 }}>
                    <label className="section-label">Analysis Pipeline</label>
                    <div className="toggle-group">
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

                  <div className="panel-section" style={{ marginTop: 14 }}>
                    <label className="section-label">Satellite Tile Selection</label>
                    <select 
                      className="custom-select"
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

                  <div className="panel-section" style={{ marginTop: 14 }}>
                    <div className="slider-header">
                      <span>Probability Threshold</span>
                      <span>{gisThreshold.toFixed(2)}</span>
                    </div>
                    <input 
                      type="range" 
                      min="0.1" 
                      max="0.9" 
                      step="0.05"
                      value={gisThreshold} 
                      onChange={(e) => setGisThreshold(parseFloat(e.target.value))}
                    />
                  </div>

                  <div className="panel-section" style={{ marginTop: 14 }}>
                    <div className="slider-header">
                      <span>Map Layer Opacity</span>
                      <span>{Math.round(gisOpacity * 100)}%</span>
                    </div>
                    <input 
                      type="range" 
                      min="0.1" 
                      max="1.0" 
                      step="0.05"
                      value={gisOpacity} 
                      onChange={(e) => setGisOpacity(parseFloat(e.target.value))}
                    />
                  </div>

                  <button 
                    className="btn btn-primary full-width" 
                    style={{ marginTop: 18 }}
                    onClick={handleRunGis}
                  >
                    <Play size={16} /> Run Spatial Analysis
                  </button>
                </div>

                {gisMetrics && (
                  <div className="gis-metrics-box">
                    <span className="m-card-title">Spatial Land Area Metrics</span>
                    <div className="gis-stat-row">
                      <span>Total Tile Area:</span>
                      <strong>{gisMetrics.total_hectares || "655.36"} ha</strong>
                    </div>
                    <div className="gis-stat-row">
                      <span>{gisMode === 'stage1' ? 'Forest Canopy:' : 'Deforestation Loss:'}</span>
                      <strong className={gisMode === 'stage1' ? 'text-green' : 'text-red'}>
                        {gisMode === 'stage1' ? `${gisMetrics.forest_hectares} ha (${gisMetrics.forest_pct}%)` : `${gisMetrics.deforestation_hectares} ha (${gisMetrics.deforestation_pct}%)`}
                      </strong>
                    </div>
                    <div className="gis-stat-row">
                      <span>Coordinates:</span>
                      <small>{gisMetrics.gis.center[0]}, {gisMetrics.gis.center[1]}</small>
                    </div>
                  </div>
                )}

                {/* Sentinel-2 Live Satellite Preview Control */}
                <div className="glass-card" style={{ marginTop: 16 }}>
                  <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 6 }}>
                    <h3 className="panel-heading" style={{ margin: 0, fontSize: '0.95rem' }}>Sentinel Preview</h3>
                    <span style={{ fontSize: '0.7rem', background: 'rgba(59,130,246,0.12)', color: '#2563eb', padding: '2px 7px', borderRadius: 4, fontWeight: 600 }}>
                      Sentinel-2 L2A
                    </span>
                  </div>
                  
                  <p style={{ fontSize: '0.78rem', color: 'var(--text-muted)', marginBottom: 12 }}>
                    Query live optical satellite imagery for Amazon Basin coordinates (-3.4653, -62.2159).
                  </p>

                  <button 
                    className="btn btn-primary full-width"
                    style={{ background: '#2563eb' }}
                    onClick={handleLoadSentinelPreview}
                    disabled={sentinelLoading}
                  >
                    {sentinelLoading ? (
                      <span>{sentinelStatusMsg || "Searching Sentinel imagery..."}</span>
                    ) : (
                      <>
                        <Globe size={16} /> Load Sentinel Preview
                      </>
                    )}
                  </button>

                  {sentinelStatusMsg && (
                    <div style={{ 
                      marginTop: 10, 
                      padding: '8px 10px', 
                      borderRadius: 6, 
                      fontSize: '0.78rem',
                      background: sentinelStatusMsg === 'Sentinel-2 imagery loaded' 
                        ? 'rgba(16,185,129,0.12)' 
                        : (sentinelLoading ? 'rgba(59,130,246,0.1)' : 'rgba(239,68,68,0.1)'),
                      color: sentinelStatusMsg === 'Sentinel-2 imagery loaded' 
                        ? '#059669' 
                        : (sentinelLoading ? '#2563eb' : '#dc2626'),
                      fontWeight: 500
                    }}>
                      {sentinelStatusMsg}
                    </div>
                  )}

                  {/* Horizontal Date Timeline / List */}
                  {sentinelAcquisitions && sentinelAcquisitions.length > 0 && (
                    <div style={{ marginTop: 14 }}>
                      <div style={{ fontSize: '0.8rem', fontWeight: 600, color: 'var(--text-main)', marginBottom: 8 }}>
                        Sentinel-2 Historical Images
                      </div>
                      <div style={{
                        display: 'flex',
                        gap: 8,
                        overflowX: 'auto',
                        paddingBottom: 6,
                        scrollbarWidth: 'thin'
                      }}>
                        {sentinelAcquisitions.map((acq, index) => {
                          const dateOnly = acq.datetime ? acq.datetime.slice(0, 10) : (acq.date || `Acq ${index + 1}`);
                          const cloudStr = acq.cloud_coverage !== undefined && acq.cloud_coverage !== null ? `${acq.cloud_coverage}%` : '';
                          const isSelected = selectedAcquisitionId === acq.id;
                          return (
                            <button
                              key={acq.id || index}
                              onClick={() => loadSentinelAcquisition(acq)}
                              disabled={sentinelLoading && isSelected}
                              style={{
                                flexShrink: 0,
                                padding: '6px 10px',
                                borderRadius: 8,
                                border: isSelected ? '2px solid #2563eb' : '1px solid var(--border-color)',
                                background: isSelected ? '#2563eb' : '#ffffff',
                                color: isSelected ? '#ffffff' : 'var(--text-main)',
                                fontSize: '0.76rem',
                                fontWeight: isSelected ? 600 : 500,
                                cursor: 'pointer',
                                transition: 'all 0.15s ease',
                                boxShadow: isSelected ? '0 2px 6px rgba(37, 99, 235, 0.25)' : 'none',
                                whiteSpace: 'nowrap'
                              }}
                              title={`Acquisition ID: ${acq.id}`}
                            >
                              {cloudStr ? `[${dateOnly} | ${cloudStr}]` : `[${dateOnly}]`}
                            </button>
                          );
                        })}
                      </div>
                    </div>
                  )}

                  {sentinelAcquisition && (
                    <div style={{ marginTop: 10, padding: '8px 10px', background: 'var(--bg-main)', borderRadius: 6, fontSize: '0.75rem', color: 'var(--text-muted)', lineHeight: 1.4 }}>
                      <div><strong>Active Acquisition:</strong></div>
                      <div><strong>Date:</strong> {sentinelAcquisition.datetime}</div>
                      {sentinelAcquisition.cloud_coverage !== undefined && (
                        <div><strong>Cloud Cover:</strong> {sentinelAcquisition.cloud_coverage}%</div>
                      )}
                      <div style={{ wordBreak: 'break-all', marginTop: 2 }}>
                        <strong>ID:</strong> {sentinelAcquisition.id}
                      </div>
                    </div>
                  )}
                </div>
              </div>

              <div className="glass-card map-container">
                <div id="react-leaflet-map" ref={mapRef}></div>
                
                <div className="map-legend-overlay">
                  <div className="legend-title">GIS Layer Legend</div>
                  <div className="legend-item">
                    <span className="legend-color legend-green"></span>
                    <span>Stage 1: Forest Canopy</span>
                  </div>
                  <div className="legend-item">
                    <span className="legend-color legend-red"></span>
                    <span>Stage 2: Deforestation Loss</span>
                  </div>
                  <div className="legend-item">
                    <span className="legend-color" style={{ background: '#3b82f6' }}></span>
                    <span>Sentinel-2 Satellite RGB</span>
                  </div>
                </div>
              </div>
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
                    <span className="stat-sub">Ground Sample Distance: 10m/px</span>
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
                  onChange={(e) => setSwipeSample(e.target.value)}
                >
                  <option value="">-- Choose Sample Pair --</option>
                  {samples.stage2_samples.map((s, i) => (
                    <option key={i} value={JSON.stringify(s)}>{s.name}</option>
                  ))}
                </select>
              </div>
              <button className="btn btn-primary" onClick={handleLoadSwipe}>
                <Play size={16} /> Load Swipe Comparer
              </button>
            </div>

            {swipeData && (
              <div className="glass-card swipe-container">
                <div className="swipe-labels">
                  <span className="label-before">◀ Before (T1 Satellite Image)</span>
                  <span className="label-after">After + AI Deforestation Overlay ▶</span>
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

                  <div className="swipe-handle" style={{ left: `${swipePos}%` }}>
                    <div className="swipe-button">↔</div>
                  </div>
                </div>
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
