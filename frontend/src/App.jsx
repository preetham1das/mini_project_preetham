import React, { useState, useEffect, useCallback, useRef } from 'react';
import {
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
  LayoutDashboard,
  Map as MapIcon,
  History as HistoryIcon,
  Cpu as CpuIcon,
  Search,
  ArrowUpRight,
  Sun,
  Moon,
  Info,
} from 'lucide-react';
import InteractiveMap from './InteractiveMap.jsx';
import EvaluationPanel from './EvaluationPanel.jsx';
import {
  getDetectionMethodPresentation,
  getDetectionWarnings,
  getForestChangeSignalPresentation,
} from './stage2Presentation.js';

export default function App() {
  const [view, setView] = useState(() => window.location.hash.replace(/^#\/?/, '') || 'dashboard');
  const [theme, setTheme] = useState(() => {
    try { return localStorage.getItem('forestwatch-theme') || 'light'; }
    catch { return 'light'; }
  });

  useEffect(() => {
    document.documentElement.dataset.theme = theme;
    try { localStorage.setItem('forestwatch-theme', theme); }
    catch { /* Theme remains active for this session when storage is unavailable. */ }
  }, [theme]);

  const [loading, setLoading] = useState(false);
  const [backendStatus, setBackendStatus] = useState('connecting');
  const [sentinelStatus, setSentinelStatus] = useState('checking');
  const liveStage2RequestRef = useRef(null);

  // Samples
  const [samples, setSamples] = useState({ stage1_samples: [], stage2_samples: [] });

  // GIS / AI overlay state (passed into the map)
  const [gisMode, setGisMode] = useState('stage1');
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

  // Separate Kaggle Amazon multimodal experiment (Sentinel-1 + Sentinel-2).
  const [kaggleS1File, setKaggleS1File] = useState(null);
  const [kaggleS2File, setKaggleS2File] = useState(null);
  const [kaggleS1RetrievedMeta, setKaggleS1RetrievedMeta] = useState(null);
  const [kaggleThreshold, setKaggleThreshold] = useState(0.5);
  const [kaggleResults, setKaggleResults] = useState(null);

  // Timeline state
  const [swipeSample, setSwipeSample] = useState('');
  const [swipeData, setSwipeData] = useState(null);
  const [timelineStep, setTimelineStep] = useState(0);

  // Benchmarks
  const [analytics, setAnalytics] = useState(null);
  const [regions, setRegions] = useState([]);
  const [latestAcquisitions, setLatestAcquisitions] = useState([]);
  const [analysisHistory, setAnalysisHistory] = useState(() => {
    try {
      const parsed = JSON.parse(localStorage.getItem('forestwatch-analysis-history') || '[]');
      return Array.isArray(parsed) ? parsed : [];
    }
    catch { return []; }
  });

  useEffect(() => {
    localStorage.setItem('forestwatch-analysis-history', JSON.stringify(analysisHistory.slice(0, 50)));
  }, [analysisHistory]);

  const recordAnalysis = useCallback((record) => {
    const entry = { id: `${Date.now()}-${Math.random()}`, analyzedAt: new Date().toISOString(), ...record };
    setAnalysisHistory((items) => [entry, ...items].slice(0, 50));
  }, []);

  const navigate = useCallback((nextPage) => {
    if (window.location.hash === `#/${nextPage}`) return;
    window.history.pushState({}, '', `#/${nextPage}`);
    setView(nextPage);
  }, []);

  useEffect(() => {
    const syncRoute = () => setView(window.location.hash.replace(/^#\/?/, '') || 'dashboard');
    window.addEventListener('popstate', syncRoute);
    window.addEventListener('hashchange', syncRoute);
    return () => {
      window.removeEventListener('popstate', syncRoute);
      window.removeEventListener('hashchange', syncRoute);
    };
  }, []);

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

    fetch('/api/gis/regions')
      .then((res) => res.json())
      .then((data) => setRegions(data.regions || []))
      .catch((err) => console.error('Regions fetch error:', err));

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

  const handleLiveStage2Result = useCallback((data, shouldRecord = true) => {
    const deforestationPct = data.deforestation_pct ?? data.deforestation_percentage;
    const deforestationPixels = data.defor_pixels ?? data.changed_pixels;
    setS2Results({
      ...data,
      pair_name: `Sentinel-2 ${data.date_before || 'T1'} → ${data.date_after || 'T2'}`,
      deforestation_pct: deforestationPct,
      deforestation_pixels: deforestationPixels,
      deforestation_hectares: data.deforestation_hectares ?? data.area_loss_hectares,
      stable_pixels: data.stable_pixels ?? (data.total_pixels - deforestationPixels),
      stable_pct: data.stable_pct ?? (100 - deforestationPct),
      t1_image: data.before_image ?? data.t1_image,
      t2_image: data.after_image ?? data.t2_image,
      acquisition_before: data.acquisition_before,
      acquisition_after: data.acquisition_after,
      live_source: true,
    });
    setGisMode('stage2');
    setAiOverlayData({ image: data.forest_change_overlay || data.forest_change_mask || data.deforestation_overlay || data.deforestation_mask, gis: data.gis, mode: 'stage2', metrics: data });
    if (shouldRecord) {
      recordAnalysis({
        type: 'Change detection',
        region: `Sentinel-2 ${data.date_before || 'T1'} → ${data.date_after || 'T2'}`,
        date: data.date_after || new Date().toISOString().slice(0, 10),
        lossHa: data.area_loss_hectares ?? data.deforestation_hectares,
        lossPct: deforestationPct,
        lossPixels: deforestationPixels,
        areaMethod: data.area_estimate_method,
      });
    }
    navigate('change');
  }, [recordAnalysis, navigate]);

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
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), 30000);

    fetch('/api/predict/stage1', { method: 'POST', body: formData, signal: controller.signal })
      .then((res) => {
        clearTimeout(timeoutId);
        return res.json();
      })
      .then((data) => {
        setLoading(false);
        if (data.error) {
          alert(data.error);
          return;
        }
        setS1Results(data);
        setGisMode('stage1');
        setAiOverlayData({ image: data.overlay_image, gis: data.gis, mode: 'stage1', metrics: data });
        recordAnalysis({ type: 'Forest analysis', region: data.sample_name || s1File?.name || 'Uploaded image', date: new Date().toISOString().slice(0, 10), forestHa: data.forest_hectares, forestPct: data.forest_pct, forestPixels: data.forest_pixels, totalPixels: data.total_pixels, areaMethod: data.area_estimate_method });
        navigate('analytics');
      })
      .catch((err) => {
        clearTimeout(timeoutId);
        setLoading(false);
        if (err.name === 'AbortError') {
          alert('Stage 1 inference timed out.');
        } else {
          alert('Backend connection error: ' + err.message);
        }
      });
  };

  const handleRunStage2 = () => {
    const formData = new FormData();
    formData.append('threshold', s2Threshold);

    if (s2FileT1 && s2FileT2) {
      formData.append('file_t1', s2FileT1);
      formData.append('file_t2', s2FileT2);
    } else if (s2Sample) {
      let sampleObj;
      try {
        sampleObj = JSON.parse(s2Sample);
      } catch {
        alert('Invalid sample selection.');
        return;
      }
      formData.append('sample_id', sampleObj.id);
      formData.append('split', sampleObj.split || 'test');
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
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), 35000);

    fetch('/api/predict/stage2', { method: 'POST', body: formData, signal: controller.signal })
      .then((res) => {
        clearTimeout(timeoutId);
        return res.json();
      })
      .then((data) => {
        setLoading(false);
        if (data.error) {
          alert(data.error);
          return;
        }
        setS2Results(data);
        setGisMode('stage2');
        setAiOverlayData({ image: data.forest_change_overlay || data.forest_change_mask || data.deforestation_overlay || data.deforestation_mask, gis: data.gis, mode: 'stage2', metrics: data });
        recordAnalysis({ type: 'Change detection', region: data.pair_name || 'Uploaded image pair', date: new Date().toISOString().slice(0, 10), lossHa: data.area_loss_hectares ?? data.deforestation_hectares, lossPct: data.deforestation_pct ?? data.deforestation_percentage, lossPixels: data.changed_pixels ?? data.deforestation_pixels, areaMethod: data.area_estimate_method });
      })
      .catch((err) => {
        clearTimeout(timeoutId);
        setLoading(false);
        if (err.name === 'AbortError') {
          alert('Stage 2 analysis timed out after 35 seconds. Please try again.');
        } else {
          alert('Backend connection error: ' + err.message);
        }
      });
  };

  const handleRunKaggleAmazon = async () => {
    if (!kaggleS1File || !kaggleS2File) {
      alert('Select both the Sentinel-1 (2-band) and Sentinel-2 (4-band) GeoTIFFs.');
      return;
    }
    const formData = new FormData();
    formData.append('sentinel1', kaggleS1File);
    formData.append('sentinel2', kaggleS2File);
    formData.append('threshold', String(kaggleThreshold));
    setLoading(true);
    try {
      const response = await fetch('/api/predict/kaggle-amazon', { method: 'POST', body: formData });
      const data = await response.json();
      if (!response.ok || data.error) throw new Error(data.error || `Request failed (${response.status})`);
      setKaggleResults(data);
      setGisMode('kaggle-amazon');
      setAiOverlayData({ image: data.map_overlay_image, gis: data.gis, mode: 'kaggle-amazon', metrics: data });
      recordAnalysis({
        type: 'Amazon multimodal inference',
        region: `${kaggleS1File.name} + ${kaggleS2File.name}`,
        date: new Date().toISOString().slice(0, 10),
        lossPct: data.deforestation_pct,
        lossPixels: data.deforestation_pixels,
        areaMethod: data.area_estimate_method,
      });
    } catch (error) {
      const isNetworkError = error instanceof TypeError && error.message === 'Failed to fetch';
      const msg = isNetworkError
        ? 'Cannot reach the inference server. Make sure the Flask backend is running on port 5000.'
        : `Kaggle Amazon inference failed: ${error.message}`;
      alert(msg);
    } finally {
      setLoading(false);
    }
  };

  const handleRetrieveKaggleSentinel1 = async (startDate, endDate) => {
    if (!kaggleS2File) throw new Error('Upload the Sentinel-2 GeoTIFF first; its footprint defines the AOI and output grid.');
    const formData = new FormData();
    formData.append('sentinel2', kaggleS2File);
    formData.append('start_date', startDate);
    formData.append('end_date', endDate);
    const response = await fetch('/api/kaggle-amazon/sentinel1', { method: 'POST', body: formData });
    if (!response.ok) {
      const data = await response.json().catch(() => ({}));
      throw new Error(data.error || `Sentinel-1 retrieval failed (${response.status})`);
    }
    const blob = await response.blob();
    const metadata = JSON.parse(response.headers.get('X-Sentinel1-Metadata') || '{}');
    const outputFile = new File([blob], 'sentinel1_VV_VH_GAMMA0_ELLIPSOID.tif', { type: 'image/tiff' });
    setKaggleS1File(outputFile);
    setKaggleS1RetrievedMeta(metadata);
    return { metadata, file: outputFile };
  };

  /* Re-run Stage 2 inference on currently loaded live Sentinel-2 images */
  const handleRunLiveStage2 = async (threshold = s2Threshold, shouldRecord = true) => {
    const beforeImg = s2Results?.before_image || s2Results?.t1_image;
    const afterImg = s2Results?.after_image || s2Results?.t2_image;
    if (!beforeImg || !afterImg) {
      alert('No live Sentinel-2 imagery available to re-run. Please select an AOI on the map.');
      return;
    }
    const dateBefore = s2Results?.date_before || 'T1';
    const dateAfter = s2Results?.date_after || 'T2';
    const bbox = s2Results?.gis?.bounds ? [
      s2Results.gis.bounds[0][1], s2Results.gis.bounds[0][0],
      s2Results.gis.bounds[1][1], s2Results.gis.bounds[1][0]
    ] : null;

    liveStage2RequestRef.current?.abort();
    const controller = new AbortController();
    liveStage2RequestRef.current = controller;
    let timedOut = false;
    const timeoutId = setTimeout(() => {
      timedOut = true;
      controller.abort();
    }, 180000);
    setLoading(true);
    try {
      const response = await fetch('/api/sentinel/analyze', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        signal: controller.signal,
        body: JSON.stringify({
          image_before: beforeImg,
          image_after: afterImg,
          date_before: dateBefore,
          date_after: dateAfter,
          acquisition_before: s2Results?.acquisition_before,
          acquisition_after: s2Results?.acquisition_after,
          detection_method: s2Results?.detection_method || 'temporal',
          bbox,
          threshold,
        }),
      });
      const data = await response.json();
      if (!response.ok || !data.success) {
        throw new Error(data.error || `Live analysis failed (HTTP ${response.status}).`);
      }
      if (liveStage2RequestRef.current === controller) {
        handleLiveStage2Result(data, shouldRecord);
      }
    } catch (error) {
      if (controller.signal.aborted) {
        if (timedOut) alert('Live analysis timed out after 3 minutes. Please try again.');
      } else {
        alert('Failed to connect to backend: ' + error.message);
      }
    } finally {
      clearTimeout(timeoutId);
      if (liveStage2RequestRef.current === controller) {
        liveStage2RequestRef.current = null;
        setLoading(false);
      }
    }
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
          label: 'Forest change',
          description: 'Red pixels show predicted canopy loss; unchanged pixels remain transparent.',
          image: swipeData.forest_change_overlay || swipeData.forest_change_mask || swipeData.deforestation_overlay,
        },
      ]
    : [];
  const selectedTimelineFrame = timelineFrames[timelineStep];

  const navItems = [
    { id: 'dashboard', label: 'Dashboard', icon: LayoutDashboard },
    { id: 'map', label: 'GIS Map', icon: MapIcon },
    { id: 'acquisition', label: 'Satellite Acquisition', icon: Satellite },
    { id: 'forest', label: 'Forest Analysis', icon: Trees },
    { id: 'change', label: 'Change Detection', icon: GitCompare },
    { id: 'kaggle-amazon', label: 'S1 + S2 Experiment', icon: Layers },
    { id: 'analytics', label: 'Analytics & Reports', icon: BarChart3 },
    { id: 'history', label: 'Timeline & History', icon: HistoryIcon },
    { id: 'benchmarks', label: 'Model Benchmarks', icon: CpuIcon },
  ];
  const page = navItems.find((item) => item.id === view) || navItems[0];
  const isMapPage = view === 'map' || view === 'acquisition';

  return (
    <div className={`app-shell fw-shell ${isMapPage ? 'fw-shell--map' : ''}`}>
      <aside className="fw-sidebar">
        <button className="fw-brand" type="button" onClick={() => navigate('dashboard')}>
          <span className="fw-brand__mark"><Radar size={19} /></span>
          <span><strong>ForestWatch</strong><small>GIS Intelligence</small></span>
        </button>
        <div className="fw-nav-label">WORKSPACE</div>
        <nav className="fw-nav" aria-label="Main navigation">
          {navItems.map(({ id, label, icon: Icon }) => (
            <button key={id} type="button" className={`fw-nav__item ${view === id ? 'is-active' : ''}`} onClick={() => navigate(id)}>
              <Icon size={16} /><span>{label}</span>
            </button>
          ))}
        </nav>
        <div className="fw-sidebar__bottom">
          <div className={`fw-health ${backendStatus}`}><i /> Inference API <strong>{backendStatus === 'online' ? 'Online' : backendStatus === 'offline' ? 'Offline' : 'Checking'}</strong></div>
          <div className={`fw-health ${sentinelStatus}`}><i /> Sentinel-2 <strong>{sentinelStatus === 'configured' ? 'Connected' : sentinelStatus === 'missing' ? 'Not configured' : 'Checking'}</strong></div>
          <small>ForestWatch GIS · {new Date().getFullYear()}</small>
        </div>
      </aside>

      <div className="fw-main">
        <header className="fw-topbar">
          <div><span className="fw-breadcrumb">ForestWatch /</span> <strong>{page.label}</strong></div>
          <div className="fw-topbar__actions">
            <div className="fw-topbar__status"><span className={`fw-status-dot ${backendStatus}`} />{backendStatus === 'online' ? 'Services operational' : backendStatus === 'offline' ? 'API unavailable' : 'Checking services'}</div>
            <button
              className="fw-theme-toggle"
              type="button"
              onClick={() => setTheme((current) => current === 'light' ? 'dark' : 'light')}
              aria-label={`Switch to ${theme === 'light' ? 'dark' : 'light'} theme`}
              title={`Switch to ${theme === 'light' ? 'dark' : 'light'} theme`}
            >
              {theme === 'light' ? <Moon size={15} /> : <Sun size={15} />}
              <span>{theme === 'light' ? 'Dark mode' : 'Light mode'}</span>
            </button>
          </div>
        </header>

        <main className={`app-main fw-content ${isMapPage ? 'fw-content--map' : ''}`}>
        {isMapPage ? (
          <div className="fw-map-page">
            <div className="fw-page-heading fw-page-heading--map">
              <div>
                {view === 'acquisition' ? <><span className="fw-eyebrow">IMAGERY WORKSPACE</span><h1>Satellite Acquisition</h1><p>Search Sentinel-2 imagery and compare two dates.</p></> : <h1>Interactive GIS Map</h1>}
              </div>
            </div>
            <InteractiveMap
              workspace={view}
              samples={samples}
              aiMode={aiOverlayData?.mode || gisMode}
              aiOpacity={gisOpacity}
              aiOverlayData={aiOverlayData}
              stage2Threshold={s2Threshold}
              onStage2Result={handleLiveStage2Result}
              onAcquisitions={setLatestAcquisitions}
            />
          </div>
        ) : view === 'forest' ? (
          <PageFrame className="fw-page--forest" title="Forest Analysis" description="Estimate forest canopy from a single satellite image.">
            <Stage1View
                  samples={samples}
                  s1Sample={s1Sample}
                  setS1Sample={setS1Sample}
                  s1File={s1File}
                  setS1File={setS1File}
                  s1Threshold={s1Threshold}
                  setS1Threshold={setS1Threshold}
                  loading={loading}
                  onRun={handleRunStage1}
                />
          </PageFrame>
        ) : view === 'change' ? (
          <PageFrame className="fw-page--change" title="Change Detection" description="Compare two dates to identify directional forest-cover change.">
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
                  onRunLive={handleRunLiveStage2}
                  onOpenMap={() => {
                    setGisMode('stage2');
                    navigate(s2Results?.live_source ? 'map' : 'acquisition');
                  }}
                />
          </PageFrame>
        ) : view === 'kaggle-amazon' ? (
          <PageFrame className="fw-page--kaggle" title="Kaggle Amazon Experiment" description="Separate Sentinel-1 SAR + Sentinel-2 optical SegNet. These inputs are sensor modalities, not a temporal before/after pair.">
            <KaggleAmazonView
              sentinel1File={kaggleS1File} setSentinel1File={setKaggleS1File}
              sentinel1RetrievedMeta={kaggleS1RetrievedMeta} setSentinel1RetrievedMeta={setKaggleS1RetrievedMeta}
              sentinel2File={kaggleS2File} setSentinel2File={setKaggleS2File}
              threshold={kaggleThreshold} setThreshold={setKaggleThreshold}
              results={kaggleResults} loading={loading} onRun={handleRunKaggleAmazon}
              onRetrieveSentinel1={handleRetrieveKaggleSentinel1}
              onViewMap={() => navigate('map')}
            />
          </PageFrame>
        ) : view === 'timeline' || view === 'history' ? (
          <PageFrame className="fw-page--history" title="Timeline & History" description="Review benchmark image pairs and analyses completed in this browser.">
            <HistoryView history={analysisHistory}>
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
            </HistoryView>
          </PageFrame>
        ) : view === 'analytics' ? (
          <PageFrame title="Analytics & Reports" description="View the latest model outputs and review saved analyses.">
            <AnalyticsView
              history={analysisHistory}
              analytics={analytics}
              s1Results={s1Results}
              s2Results={s2Results}
              overlay={aiOverlayData}
              overlayOpacity={gisOpacity}
              onOverlayOpacityChange={setGisOpacity}
              onViewMap={() => navigate('map')}
            />
          </PageFrame>
        ) : view === 'benchmarks' ? (
          <PageFrame title="Model Benchmarks" description="Evaluation results from the recorded benchmark datasets. Live scene results are shown separately from these scores.">
            <EvaluationPanel analytics={analytics} />
          </PageFrame>
        ) : (
          <DashboardView
            history={analysisHistory}
            regions={regions}
            acquisitions={latestAcquisitions}
            backendStatus={backendStatus}
            sentinelStatus={sentinelStatus}
            latestForest={s1Results}
            analytics={analytics}
            onNavigate={navigate}
          />
        )}
      </main>
      </div>
    </div>
  );
}

function PageFrame({ title, description, children, className = '' }) {
  return (
    <div className={`fw-page view ${className}`}>
      <div className="fw-page-heading">
        <div><span className="fw-eyebrow">FORESTWATCH GIS</span><h1>{title}</h1><p>{description}</p></div>
      </div>
      <div className="fw-page-scroll">{children}</div>
    </div>
  );
}

function DashboardView({ history, regions, acquisitions, backendStatus, sentinelStatus, latestForest, analytics, onNavigate }) {
  const latestLoss = history.find((item) => item.lossPct != null);
  const latestCanopy = latestForest || history.find((item) => item.forestPct != null);
  const stage1Iou = analytics?.stage1?.iou;
  const cards = [
    { label: 'Georeferenced regions', value: regions.length || '—', detail: regions.length ? 'Spatial metadata available' : 'No bundled sample coordinates', icon: MapPin, tone: 'green' },
    { label: 'Completed analyses', value: history.length, detail: 'Saved in this browser', icon: Activity, tone: 'cyan' },
    { label: 'Latest detected loss', value: latestLoss ? `${latestLoss.lossPct}%` : '—', detail: latestLoss ? `${latestLoss.lossPixels?.toLocaleString() ?? '—'} pixels · ${latestLoss.region}` : 'Run a change analysis to populate', icon: Trees, tone: 'red' },
    { label: 'Latest canopy cover', value: latestCanopy ? `${latestCanopy.forest_pct ?? latestCanopy.forestPct}%` : '—', detail: latestCanopy ? `${latestCanopy.forestPixels?.toLocaleString() ?? '—'} pixels · ${latestCanopy.region || 'latest analysis'}` : 'Run forest analysis to populate', icon: ScanLine, tone: 'green' },
  ];
  return (
    <div className="fw-page view">
      <div className="fw-page-heading"><div><span className="fw-eyebrow">FOREST INTELLIGENCE</span><h1>Dashboard</h1><p>Monitor imagery services and review the work completed in this browser.</p></div><button className="btn btn--primary" onClick={() => onNavigate('acquisition')}><Satellite size={14} /> Start an analysis</button></div>
      <div className="fw-page-scroll">
        <section className="fw-stat-grid">
          {cards.map(({ label, value, detail, icon: Icon, tone }) => <article key={label} className="fw-stat-card"><div className={`fw-stat-icon ${tone}`}><Icon size={17} /></div><span>{label}</span><strong>{value}</strong><small>{detail}</small></article>)}
        </section>
        <div className="fw-dashboard-grid">
          <section className="fw-card fw-activity-card"><div className="fw-section-head"><div><h2>Recent analysis activity</h2><p>Results recorded locally in this browser</p></div><button className="fw-text-button" onClick={() => onNavigate('history')}>View history <ArrowUpRight size={14} /></button></div>
            {history.length ? <div className="fw-activity-list">{history.slice(0, 5).map((item) => {
              const measuredResult = item.lossPct != null
                ? `${item.lossHa != null ? `${item.lossHa} ha · ` : ''}${item.lossPct}% loss`
                : item.forestPct != null
                  ? `${item.forestHa != null ? `${item.forestHa} ha · ` : ''}${item.forestPct}% canopy`
                  : 'Completed';
              return <div className="fw-activity-row" key={item.id}><span className={`fw-activity-mark ${item.type === 'Forest analysis' ? 'forest' : 'loss'}`} /><div><strong>{item.type}</strong><small>{item.region || 'Analysis'} · {item.date || new Date(item.analyzedAt).toLocaleDateString()}</small></div><b>{measuredResult}</b></div>;
            })}</div> : <EmptyState title="No completed analyses yet" text="Run a Forest U-Net or forest-loss workflow to see results here." />}
          </section>
          <section className="fw-card"><div className="fw-section-head"><div><h2>Service health</h2><p>Connectivity reported by the backend</p></div><ShieldCheck size={17} /></div>
            <ServiceRow label="Inference API" status={backendStatus === 'online' ? 'Online' : backendStatus === 'offline' ? 'Offline' : 'Checking'} tone={backendStatus} />
            <ServiceRow label="Sentinel-2 STAC" status={sentinelStatus === 'configured' ? 'Connected' : sentinelStatus === 'missing' ? 'Credentials missing' : 'Checking'} tone={sentinelStatus} />
            <div className="fw-model-note"><CpuIcon size={16} /><div><strong>Model benchmarks</strong><span>{stage1Iou ? `Forest U-Net IoU ${(stage1Iou * 100).toFixed(1)}%` : 'See recorded benchmark results'}</span></div><button aria-label="Open model benchmarks" onClick={() => onNavigate('benchmarks')}><ChevronRight size={15} /></button></div>
          </section>
          <section className="fw-card fw-regions-card"><div className="fw-section-head"><div><h2>Georeferenced regions</h2><p>Locations with verified spatial metadata</p></div><button className="fw-text-button" onClick={() => onNavigate('map')}>Open map <ArrowUpRight size={14} /></button></div>
            {regions.length ? <div className="fw-region-list">{regions.slice(0, 4).map((region) => <div className="fw-region-row" key={region.id}><MapPin size={14} /><span><strong>{region.region_name || region.name}</strong><small>{region.center?.map((n) => Number(n).toFixed(3)).join(', ') || 'Coordinates unavailable'}</small></span><span className="fw-region-gsd">{region.gsd_meters || '—'} m</span></div>)}</div> : <EmptyState title="No sample coordinates available" text="Bundled images have no location metadata. Draw a real AOI on the map to analyze a georeferenced area." />}
          </section>
          <section className="fw-card"><div className="fw-section-head"><div><h2>Latest satellite acquisitions</h2><p>From the most recent Sentinel-2 search</p></div><button className="fw-text-button" onClick={() => onNavigate('acquisition')}>Search scenes <ArrowUpRight size={14} /></button></div>
            {acquisitions?.length ? <div className="fw-region-list">{acquisitions.slice(-4).reverse().map((scene) => <div className="fw-region-row" key={scene.id}><Satellite size={14} /><span><strong>{scene.datetime ? new Date(scene.datetime).toLocaleString() : scene.id}</strong><small>{scene.id}</small></span><span className="fw-region-gsd">{scene.cloud_coverage == null ? 'Cloud n/a' : `${scene.cloud_coverage}% cloud`}</span></div>)}</div> : <EmptyState title="No recent acquisitions" text="Search for Sentinel-2 scenes on the acquisition page." />}
          </section>
          <section className="fw-card fw-trend-card"><div className="fw-section-head"><div><h2>Forest loss trend</h2><p>Share of analyzed pixels classified as loss</p></div><BarChart3 size={17} /></div><TrendChart metric="lossPct" unit="%" records={history.filter((item) => item.lossPct != null && Number.isFinite(Number(item.lossPct))).slice(0, 8).reverse()} /></section>
          <section className="fw-card fw-trend-card"><div className="fw-section-head"><div><h2>Forest cover trend</h2><p>Canopy estimates from completed Forest U-Net runs</p></div><Trees size={17} /></div><TrendChart metric="forestPct" unit="%" records={history.filter((item) => Number.isFinite(Number(item.forestPct))).slice(0, 8).reverse()} /></section>
        </div>
      </div>
    </div>
  );
}

function ServiceRow({ label, status, tone }) {
  const ok = tone === 'online' || tone === 'configured';
  const warn = tone === 'connecting' || tone === 'checking';
  return <div className="fw-service-row"><span><i className={ok ? 'ok' : warn ? 'warn' : 'off'} />{label}</span><strong>{status}</strong></div>;
}

function OverlaySettings({ overlay, opacity, onOpacityChange, onViewMap }) {
  return (
    <section className="fw-card fw-overlay-settings">
      <div className="fw-section-head">
        <div><h2>Map overlay</h2><p>Control how the latest inference result appears on the GIS map.</p></div>
        <span className={`fw-overlay-status ${overlay?.gis?.bounds ? 'is-active' : ''}`}><i />{overlay?.gis?.bounds ? 'Ready' : overlay ? 'No map coordinates' : 'No result yet'}</span>
      </div>
      <div className="fw-overlay-controls">
        <label htmlFor="result-overlay-opacity">Overlay opacity <strong>{Math.round(opacity * 100)}%</strong></label>
        <input id="result-overlay-opacity" type="range" min="0" max="1" step="0.05" value={opacity} onChange={(event) => onOpacityChange(Number(event.target.value))} />
        <span className="fw-overlay-hint">{overlay?.gis?.bounds ? `${overlay.mode === 'stage1' ? 'Forest canopy' : 'Forest loss'} result has AOI coordinates and can be placed on the map.` : overlay ? 'This image has no verified geographic coordinates; review the mask and overlay above.' : 'Run an analysis to generate a map overlay.'}</span>
      </div>
      <button type="button" className="btn btn--primary" onClick={onViewMap} disabled={!overlay?.gis?.bounds}>View result on GIS map</button>
    </section>
  );
}

function EmptyState({ title, text }) {
  return <div className="fw-empty"><Search size={17} /><strong>{title}</strong><span>{text}</span></div>;
}

function TrendChart({ records, metric = 'lossHa', unit = 'ha' }) {
  if (!records.length) return <EmptyState title="Trend data will appear here" text={unit === '%' ? 'No pixel-percentage results have been recorded yet.' : 'No georeferenced area measurements have been recorded yet.'} />;
  const max = Math.max(...records.map((record) => Number(record[metric])), 1);
  return <div className="fw-chart" role="img" aria-label={unit === '%' ? 'Forest cover percentage by analysis' : 'Forest loss hectares by analysis'}><div className="fw-chart__bars">{records.map((record, index) => <div className="fw-chart__col" key={record.id || index}><span>{Number(record[metric]).toLocaleString()} {unit}</span><i style={{ height: `${Math.max(6, Number(record[metric]) / max * 100)}%` }} /><small>{record.date || 'Analysis'}</small></div>)}</div></div>;
}

function AnalyticsView({ history, analytics, s1Results, s2Results, overlay, overlayOpacity, onOverlayOpacityChange, onViewMap }) {
  const lossRecords = history.filter((item) => item.lossPct != null && Number.isFinite(Number(item.lossPct)));
  const exportCsv = () => {
    const rows = [['Analysis date', 'Type', 'Region', 'Forest loss (ha estimate)', 'Forest loss (%)', 'Forest loss (pixels)', 'Forest cover (ha estimate)', 'Forest cover (%)', 'Forest cover (pixels)', 'Total pixels', 'Area estimate method'], ...history.map((item) => [item.date || '', item.type || '', item.region || '', item.areaMethod ? item.lossHa ?? '' : '', item.lossPct ?? '', item.lossPixels ?? '', item.areaMethod ? item.forestHa ?? '' : '', item.forestPct ?? '', item.forestPixels ?? '', item.totalPixels ?? '', item.areaMethod || ''])];
    const csv = rows.map((row) => row.map((cell) => `"${String(cell).replaceAll('"', '""')}"`).join(',')).join('\r\n');
    const url = URL.createObjectURL(new Blob([csv], { type: 'text/csv;charset=utf-8' }));
    const link = document.createElement('a'); link.href = url; link.download = 'forestwatch-analysis-history.csv'; link.click(); URL.revokeObjectURL(url);
  };
  const benchmarkSources = analytics?.stage1 && analytics?.stage2 ? '2 models' : analytics?.stage1 ? 'Forest U-Net' : analytics?.stage2 ? 'Temporal U-Net' : 'Unavailable';
  const forestRecords = history.filter((item) => Number.isFinite(Number(item.forestPct)));
  return (
    <div className="fw-reports">
      {s1Results && <LatestForestResults results={s1Results} />}
      {s2Results && <LatestChangeResults results={s2Results} />}
      {overlay && <OverlaySettings overlay={overlay} opacity={overlayOpacity} onOpacityChange={onOverlayOpacityChange} onViewMap={onViewMap} />}
      <section className="fw-card">
        <div className="fw-section-head"><div><h2>Analysis report</h2><p>Export pixel metrics and any AOI-based area estimates recorded in this browser.</p></div><button className="btn btn--ghost" onClick={exportCsv} disabled={!history.length}>Download CSV</button></div>
        <div className="fw-report-kpis"><div><span>Saved analyses</span><strong>{history.length}</strong></div><div><span>Change analyses</span><strong>{history.filter((item) => item.type === 'Change detection' || item.type === 'Amazon multimodal inference').length}</strong></div><div><span>Loss percentage records</span><strong>{lossRecords.length}</strong></div><div><span>Benchmark sources</span><strong>{benchmarkSources}</strong></div></div>
      </section>
      <section className="fw-card"><div className="fw-section-head"><div><h2>Forest loss by analysis</h2><p>Percentage of analyzed pixels classified as loss</p></div></div><TrendChart metric="lossPct" unit="%" records={lossRecords.slice(0, 12).reverse()} /></section>
      <section className="fw-card"><div className="fw-section-head"><div><h2>Canopy cover by analysis</h2><p>Coverage reported by Forest U-Net runs</p></div></div><TrendChart metric="forestPct" unit="%" records={forestRecords.slice(0, 12).reverse()} /></section>
      <section className="fw-card"><div className="fw-section-head"><div><h2>Analysis records</h2><p>Pixel counts and any AOI-based area estimates</p></div></div><HistoryTable history={history} /></section>
    </div>
  );
}

function HistoryView({ history, children }) {
  return <div className="fw-history-layout">{children}<section className="fw-card fw-history-card"><div className="fw-section-head"><div><h2>Analysis history</h2><p>Saved in this browser only; records are not synced to a server.</p></div><span className="fw-history-count">{history.length} {history.length === 1 ? 'record' : 'records'}</span></div><HistoryTable history={history} /></section></div>;
}

function HistoryTable({ history }) {
  if (!history.length) return <EmptyState title="No analysis history" text="Completed analyses will be listed here." />;
  return <div className="fw-table-wrap"><table className="fw-table"><thead><tr><th>Analysis</th><th>Region / pair</th><th>Date</th><th>Measured result</th></tr></thead><tbody>{history.map((item) => { const result = item.lossPct != null ? `${item.areaMethod && item.lossHa != null ? `${item.lossHa} ha estimate · ` : ''}${item.lossPct}% loss${item.lossPixels != null ? ` · ${Number(item.lossPixels).toLocaleString()} px` : ''}` : item.forestPct != null ? `${item.areaMethod && item.forestHa != null ? `${item.forestHa} ha estimate · ` : ''}${item.forestPct}% cover${item.forestPixels != null ? ` · ${Number(item.forestPixels).toLocaleString()} px` : ''}` : '—'; return <tr key={item.id}><td>{item.type}</td><td>{item.region || '—'}</td><td>{item.date || new Date(item.analyzedAt).toLocaleDateString()}</td><td>{result}</td></tr>; })}</tbody></table></div>;
}

/* ══════════════════════════════════════════════════════════════
   Manual workbench views
   ══════════════════════════════════════════════════════════════ */

function KaggleAmazonView({ sentinel1File, setSentinel1File, sentinel1RetrievedMeta, setSentinel1RetrievedMeta,
  sentinel2File, setSentinel2File, threshold, setThreshold, results, loading,
  onRun, onRetrieveSentinel1, onViewMap }) {
  const [researchMetrics, setResearchMetrics] = useState(null);
  const [researchMetricsError, setResearchMetricsError] = useState('');
  const [dateRange, setDateRange] = useState(() => {
    const end = new Date();
    const start = new Date(end); start.setDate(start.getDate() - 30);
    return { start: start.toISOString().slice(0, 10), end: end.toISOString().slice(0, 10) };
  });
  const [retrieving, setRetrieving] = useState(false);
  const [retrieveError, setRetrieveError] = useState('');
  const [sentinel1DownloadUrl, setSentinel1DownloadUrl] = useState('');

  useEffect(() => () => { if (sentinel1DownloadUrl) URL.revokeObjectURL(sentinel1DownloadUrl); }, [sentinel1DownloadUrl]);

  useEffect(() => {
    fetch('/api/kaggle-amazon/research-metrics')
      .then((response) => {
        if (!response.ok) throw new Error(`Research metrics unavailable (${response.status})`);
        return response.json();
      })
      .then(setResearchMetrics)
      .catch((error) => setResearchMetricsError(error.message));
  }, []);

  const retrieveSentinel1 = async () => {
    setRetrieving(true); setRetrieveError('');
    try {
      const { metadata, file } = await onRetrieveSentinel1(dateRange.start, dateRange.end);
      setSentinel1DownloadUrl((old) => { if (old) URL.revokeObjectURL(old); return URL.createObjectURL(file); });
      if (!metadata?.calibration_compatible_with_kaggle_v2) setRetrieveError('Retrieved successfully. The V2 training files do not record the Sentinel-1 backscatter coefficient or units, and their saved ranges do not match linear power. This file is available to download, but inference is disabled until calibration is verified.');
    } catch (error) {
      setRetrieveError(error.message || 'Sentinel-1 retrieval failed.');
    } finally {
      setRetrieving(false);
    }
  };

  return (
    <>
      <section className="panel kaggle-research-panel">
        <div className="panel__head">
          <div>
            <div className="panel__title">Recorded research evaluation</div>
            <div className="block__hint">V2 Colab held-out test results · separate from the live result below</div>
          </div>
          <span className="badge badge--green"><ShieldCheck size={11} /> Threshold 0.50</span>
        </div>
        {researchMetrics ? (
          <>
            <div className="stat-grid kaggle-research-stats">
              {[
                ['IoU', researchMetrics.held_out_test.iou],
                ['Dice / F1', researchMetrics.held_out_test.dice_f1],
                ['Precision', researchMetrics.held_out_test.precision],
                ['Recall', researchMetrics.held_out_test.recall],
                ['Accuracy', researchMetrics.held_out_test.accuracy],
              ].map(([label, value]) => (
                <div className="stat stat--cyan" key={label}>
                  <span className="stat__label">{label}</span>
                  <strong className="stat__value">{Number(value).toFixed(4)}</strong>
                </div>
              ))}
            </div>
            <div className="kaggle-research-meta">
              <span>Best validation epoch: {researchMetrics.best_validation.epoch}</span>
              <span>Training epochs: {researchMetrics.training.epochs_completed}</span>
              <span>Confusion matrix [[TN, FP], [FN, TP]]: {researchMetrics.held_out_test.confusion_matrix.map((row) => row.join(', ')).join('] [')}</span>
            </div>
            <div className="block__hint kaggle-audit-note">{researchMetrics.audit_note}</div>
          </>
        ) : <div className="block__hint">{researchMetricsError || 'Loading recorded metrics…'}</div>}
      </section>

      <section className="panel">
        <div className="panel__head">
          <div>
            <div className="panel__title">Sentinel-1 + Sentinel-2 late-fusion SegNet</div>
            <div className="block__hint">This separate experiment requires aligned multi-band GeoTIFFs: 2 SAR bands and 4 optical bands.</div>
          </div>
          <span className="badge badge--green"><Layers size={11} /> Kaggle Amazon V2</span>
        </div>
        <div className="block__hint" style={{ marginBottom: 14 }}>
          Choose co-registered rasters with matching dimensions, CRS and affine transform. This model uses the V2 per-band normalization saved with its checkpoint. PNG/JPEG and ordinary RGB images are not supported.
        </div>
        <div className="control-grid">
          <label className={`upload-zone ${sentinel1File ? 'has-file' : ''}`}>
            <input type="file" accept=".tif,.tiff,image/tiff" onChange={(event) => { setSentinel1File(event.target.files?.[0] || null); setSentinel1RetrievedMeta(null); setRetrieveError(''); }} />
            <Upload size={17} />
            <span className="upload-zone__label">{sentinel1File?.name || 'Upload Sentinel-1 GeoTIFF'}</span>
            <span className="upload-zone__hint">2 bands · same grid as Sentinel-2</span>
          </label>
          <label className={`upload-zone ${sentinel2File ? 'has-file' : ''}`}>
            <input type="file" accept=".tif,.tiff,image/tiff" onChange={(event) => { const nextFile = event.target.files?.[0] || null; setSentinel2File(nextFile); if (sentinel1RetrievedMeta) { setSentinel1File(null); setSentinel1RetrievedMeta(null); setSentinel1DownloadUrl(''); } }} />
            <Upload size={17} />
            <span className="upload-zone__label">{sentinel2File?.name || 'Upload Sentinel-2 GeoTIFF'}</span>
            <span className="upload-zone__hint">4 bands · same grid as Sentinel-1</span>
          </label>
          <div className="slider-block">
            <div className="slider-row"><span>Deforestation threshold</span><span className="slider-value">{threshold.toFixed(2)}</span></div>
            <input type="range" min="0.1" max="0.9" step="0.05" value={threshold} onChange={(event) => setThreshold(Number(event.target.value))} aria-label="Kaggle model threshold" />
          </div>
          <button type="button" className="btn btn--primary" onClick={onRun} disabled={loading || !sentinel1File || !sentinel2File || sentinel1RetrievedMeta?.calibration_compatible_with_kaggle_v2 === false}>
            {loading ? <span className="btn__spinner" /> : <Play size={14} />}
            Run Kaggle model
          </button>
        </div>
        <div className="kaggle-retrieve">
          <div className="kaggle-retrieve__copy">
            <strong><Radar size={14} /> Retrieve Sentinel-1 VV + VH</strong>
            <span>Uses the uploaded Sentinel-2 GeoTIFF footprint and grid as the AOI/reference.</span>
          </div>
          <div className="kaggle-retrieve__fields">
            <label>From<input type="date" value={dateRange.start} max={dateRange.end} onChange={(event) => setDateRange((current) => ({ ...current, start: event.target.value }))} /></label>
            <label>To<input type="date" value={dateRange.end} min={dateRange.start} onChange={(event) => setDateRange((current) => ({ ...current, end: event.target.value }))} /></label>
            <button type="button" className="btn btn--ghost" onClick={retrieveSentinel1} disabled={retrieving || !sentinel2File || !dateRange.start || !dateRange.end}>
              {retrieving ? <span className="btn__spinner" /> : <RefreshCw size={14} />}
              {retrieving ? 'Retrieving…' : 'Find and download VV/VH'}
            </button>
          </div>
          {sentinel1RetrievedMeta && <div className="kaggle-retrieve__meta">{sentinel1RetrievedMeta.width} × {sentinel1RetrievedMeta.height} · {sentinel1RetrievedMeta.crs} · {sentinel1RetrievedMeta.catalog_item_count} matching catalogue item(s) · aligned to Sentinel-2</div>}
          {sentinel1RetrievedMeta?.acquisition_dates?.length > 0 && <div className="kaggle-retrieve__meta">Acquisition dates: {sentinel1RetrievedMeta.acquisition_dates.map((date) => new Date(date).toLocaleDateString()).join(', ')}</div>}
          {sentinel1RetrievedMeta && sentinel1DownloadUrl && <a className="btn btn--ghost kaggle-retrieve__download" href={sentinel1DownloadUrl} download="sentinel1_VV_VH_GAMMA0_ELLIPSOID.tif">Download numeric VV/VH GeoTIFF</a>}
          {retrieveError && <div className={sentinel1RetrievedMeta ? 'kaggle-calibration-warning' : 'kaggle-retrieve__error'}>{retrieveError}</div>}
        </div>
        <div className="kaggle-calibration-warning">
          The V2 checkpoint was trained on raw Kaggle raster values with saved train-only ranges. That source does not identify its Sentinel-1 coefficient or units. Sentinel Hub returns calibrated linear-power data, so retrieved files are kept out of model inference until the training calibration can be confirmed.
        </div>
      </section>
      {results && (
        <>
          <div className="stat-grid" style={{ gridTemplateColumns: 'repeat(auto-fit, minmax(190px, 1fr))' }}>
            <div className="stat stat--red"><span className="stat__label">Predicted deforestation</span><strong className="stat__value">{results.deforestation_pct}%</strong><small className="stat__sub">{Number(results.deforestation_pixels).toLocaleString()} pixels</small></div>
            <div className="stat stat--cyan"><span className="stat__label">Valid input pixels</span><strong className="stat__value">{Number(results.valid_pixels).toLocaleString()}</strong><small className="stat__sub">{results.width} × {results.height} · {results.crs}</small></div>
            {results.deforestation_hectares != null && <div className="stat stat--green"><span className="stat__label">Estimated area</span><strong className="stat__value">{results.deforestation_hectares} ha</strong><small className="stat__sub">Calculated from GeoTIFF pixel size and projected CRS</small></div>}
          </div>
          <section className="panel">
            <div className="panel__head"><div><div className="panel__title">Kaggle model output</div><div className="block__hint">Threshold {results.threshold} · Preview is downsampled for display.</div></div></div>
            <div className="kaggle-result-actions">
              <button type="button" className="btn btn--ghost" onClick={onViewMap} disabled={!results.gis?.bounds}>Show overlay on GIS map</button>
              <a className="btn btn--ghost" href={results.download_url} download="amazon_deforestation_prediction.tif">Download georeferenced mask</a>
            </div>
            <div className="visuals-grid">
              <figure><img src={results.preview_image} alt="Sentinel-2 input preview" /><figcaption>Sentinel-2 preview</figcaption></figure>
              <figure><img src={results.prediction_mask} alt="Predicted deforestation mask" /><figcaption>Predicted mask</figcaption></figure>
              <figure><img src={results.overlay_image} alt="Predicted mask overlay" /><figcaption>Overlay</figcaption></figure>
            </div>
          </section>
        </>
      )}
    </>
  );
}

function Stage1View({
  samples,
  s1Sample,
  setS1Sample,
  s1File,
  setS1File,
  s1Threshold,
  setS1Threshold,
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

        <div className="control-grid fw-forest-controls">
          <div className="fw-forest-source">
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
            <button type="button" className="btn btn--primary" onClick={onRun} disabled={loading || (!s1Sample && !s1File)}>
              {loading ? <span className="btn__spinner" /> : <Play size={14} />}
              Run forest analysis
            </button>
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

        </div>
      </section>

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
  onRunLive,
  onOpenMap,
}) {
  const liveThresholdTimerRef = useRef(null);
  const [stage2Mode, setStage2Mode] = useState(
    s2Results?.live_source ? 'live' : (s2FileT1 || s2FileT2 ? 'upload' : s2Sample ? 'sample' : 'live')
  );
  const resultMethod = getDetectionMethodPresentation(s2Results);

  useEffect(() => () => clearTimeout(liveThresholdTimerRef.current), []);

  return (
    <>
      <section className="panel fw-change-panel">
        <div className="panel__head">
          <div>
            <div className="panel__title">Bi-Temporal Deforestation Detection</div>
            <div className="block__hint">
              {s2Results ? resultMethod.description : 'Run an analysis to see which model produced its result.'}
            </div>
          </div>
          <span className="badge badge--violet">
            <Layers size={11} /> {s2Results ? resultMethod.label : resultMethod.shortLabel}
          </span>
        </div>

        {/* ── Mode Selector: Clean, Segmented Switcher ── */}
        <div className="seg fw-mode-switcher" role="tablist" aria-label="Change detection source">
          <button
            type="button"
            role="tab"
            aria-selected={stage2Mode === 'live'}
            className={`seg__btn ${stage2Mode === 'live' ? 'is-active' : ''}`}
            style={{ flex: 1, justifyContent: 'center' }}
            onClick={() => setStage2Mode('live')}
          >
            <Satellite size={12} /> Live Sentinel-2
          </button>
          <button
            type="button"
            role="tab"
            aria-selected={stage2Mode === 'sample'}
            className={`seg__btn ${stage2Mode === 'sample' ? 'is-active' : ''}`}
            style={{ flex: 1, justifyContent: 'center' }}
            onClick={() => setStage2Mode('sample')}
          >
            <GitCompare size={12} /> Sample Image Pair
          </button>
          <button
            type="button"
            role="tab"
            aria-selected={stage2Mode === 'upload'}
            className={`seg__btn ${stage2Mode === 'upload' ? 'is-active' : ''}`}
            style={{ flex: 1, justifyContent: 'center' }}
            onClick={() => setStage2Mode('upload')}
          >
            <Upload size={12} /> Upload Image Pair
          </button>
        </div>

        {/* ── Mode A: Live Sentinel-2 ── */}
        {stage2Mode === 'live' && (
          <div className="control-grid fw-change-live">
            {sentinelStatus !== 'configured' && (
              <div className="note note--warn">
                <RefreshCw size={12} />
                <span>
                  {sentinelStatus === 'checking'
                    ? 'Checking Sentinel-2 STAC service configuration…'
                    : sentinelStatus === 'missing'
                      ? 'Live imagery is unavailable until SENTINEL_CLIENT_ID and SENTINEL_CLIENT_SECRET are configured. Prepared sample pairs and offline uploads remain active.'
                      : 'Sentinel-2 status could not be verified. Confirm credentials in .env.'}
                </span>
              </div>
            )}

            {s2Results?.live_source ? (
              <div className="fw-change-live__content">
                <div className="note note--ok">
                  <Satellite size={13} />
                  <span>
                    <strong>Active Live Pair:</strong> {s2Results.pair_name || 'Sentinel-2 Scene'} · {resultMethod.shortLabel}
                  </span>
                </div>

                <div className="slider-block">
                  <div className="slider-row">
                    <span>Deforestation confidence threshold</span>
                    <span className="slider-value">{s2Threshold.toFixed(2)}</span>
                  </div>
                  <input
                    type="range"
                    min="0.2"
                    max="0.7"
                    step="0.05"
                    value={s2Threshold}
                    onChange={(e) => {
                      const nextThreshold = parseFloat(e.target.value);
                      setS2Threshold(nextThreshold);
                      clearTimeout(liveThresholdTimerRef.current);
                      liveThresholdTimerRef.current = setTimeout(
                        () => onRunLive?.(nextThreshold, false),
                        300
                      );
                    }}
                    aria-label="Deforestation threshold"
                  />
                </div>

                <div className="row" style={{ gap: 8, marginTop: 4 }}>
                  <button
                    type="button"
                    className="btn btn--primary"
                    style={{ flex: 1 }}
                    onClick={() => onRunLive?.(s2Threshold)}
                    disabled={loading}
                  >
                    {loading ? <span className="btn__spinner" /> : <Play size={14} />}
                    Re-run Detection
                  </button>
                  <button
                    type="button"
                    className="btn btn--cyan"
                    onClick={onOpenMap}
                  >
                    <MapPin size={13} /> Choose New AOI on Map
                  </button>
                </div>
              </div>
            ) : (
              <div className="fw-change-live__content">
                <div className="note note--info">
                  <Satellite size={13} />
                  <span>
                    Choose an AOI and two dates on the GIS Map. Loss is T1 forest that is not forest at T2.
                  </span>
                </div>

                <div className="row" style={{ justifyContent: 'flex-start', marginTop: 4 }}>
                  <button type="button" className="btn btn--cyan" onClick={onOpenMap}>
                    <MapPin size={14} /> Open GIS Map &amp; Draw AOI
                  </button>
                </div>
              </div>
            )}
          </div>
        )}

        {/* ── Mode B: Sample Image Pair ── */}
        {stage2Mode === 'sample' && (
          <div className="control-grid fw-change-controls fw-change-controls--sample">
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
                <option value="">— Choose a benchmark sample pair —</option>
                {samples.stage2_samples.map((s, i) => (
                  <option key={i} value={JSON.stringify(s)}>
                    {s.name}
                  </option>
                ))}
              </select>
            </div>

            <div className="slider-block fw-change-threshold">
              <div className="slider-row">
                <span>Deforestation threshold</span>
                <span className="slider-value">{s2Threshold.toFixed(2)}</span>
              </div>
              <input
                type="range"
                min="0.2"
                max="0.7"
                step="0.05"
                value={s2Threshold}
                onChange={(e) => setS2Threshold(parseFloat(e.target.value))}
                aria-label="Deforestation threshold"
              />
            </div>

            <button
              type="button"
              className="btn btn--primary fw-change-submit"
              onClick={onRun}
              disabled={loading || !s2Sample}
            >
              {loading ? <span className="btn__spinner" /> : <Play size={14} />}
              Detect Deforestation
            </button>
          </div>
        )}

        {/* ── Mode C: Upload Image Pair ── */}
        {stage2Mode === 'upload' && (
          <div className="control-grid fw-change-controls fw-change-controls--upload">
            <div className="block__hint fw-change-note">
              Upload two co-registered satellite images covering the exact same geographic extent (PNG, TIFF, JPG). The upload endpoint checks matching pixel dimensions; it cannot verify geospatial coordinates or CRS, so confirm both files are spatially aligned before running.
            </div>

            <label className={`upload-zone fw-change-upload ${s2FileT1 ? 'has-file' : ''}`}>
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
              <span className="upload-zone__hint">Earlier timestamp</span>
            </label>

            <label className={`upload-zone fw-change-upload ${s2FileT2 ? 'has-file' : ''}`}>
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
              <span className="upload-zone__hint">Later timestamp</span>
            </label>

            <div className="slider-block fw-change-threshold">
              <div className="slider-row">
                <span>Deforestation threshold</span>
                <span className="slider-value">{s2Threshold.toFixed(2)}</span>
              </div>
              <input
                type="range"
                min="0.2"
                max="0.7"
                step="0.05"
                value={s2Threshold}
                onChange={(e) => setS2Threshold(parseFloat(e.target.value))}
                aria-label="Deforestation threshold"
              />
            </div>

            <button
              type="button"
              className="btn btn--primary fw-change-submit"
              onClick={onRun}
              disabled={loading || !s2FileT1 || !s2FileT2}
            >
              {loading ? <span className="btn__spinner" /> : <Play size={14} />}
              Detect Deforestation
            </button>
          </div>
        )}
      </section>
      {s2Results && <LatestChangeResults results={s2Results} />}
    </>
  );
}

function LatestChangeResults({ results }) {
  if (!results) return null;
  const methodPresentation = getDetectionMethodPresentation(results);
  const signalPresentation = getForestChangeSignalPresentation(results.signal);
  const signalIsLoss = results.signal === 'DEFORESTATION';
  const warnings = getDetectionWarnings(results);
  const areaUnit = results.area_unit === 'ha' || results.deforestation_hectares != null ? 'ha' : 'px';
  const activeArea = results.deforestation_hectares ?? results.deforested_area ?? results.deforested_pixels ?? results.deforestation_pixels ?? results.defor_pixels ?? 0;
  const activePixels = results.deforested_pixels ?? results.deforestation_pixels ?? results.defor_pixels ?? 0;
  return (
    <section className="fw-latest-results">
      <div className="fw-section-head">
        <div><h2>Latest change detection</h2><p>{results.pair_name || 'Most recent temporal image pair'}</p></div>
        {signalPresentation && <span className={`fw-results-badge fw-results-badge--${signalPresentation.tone}`}>{signalPresentation.label}</span>}
        {results.live_source && <span className="fw-results-badge">Live imagery · no ground-truth evaluation</span>}
        <span className="fw-results-badge">{methodPresentation.label} · {results.model_status || 'complete'}</span>
      </div>
      <div className="stat-grid fw-change-results">
        {signalPresentation && <div className={`stat stat--${signalIsLoss ? 'red' : 'neutral'}`}>
          <span className="stat__label">{results.status || signalPresentation.label}</span>
          <div className="stat__value">
            {formatChangeArea(activeArea, areaUnit)}
            <span className="stat__unit">{areaUnit}</span>
          </div>
          <div className="stat__sub">{Number(activePixels).toLocaleString()} deforestation pixels{s2AreaEstimate(results) ? ' · AOI estimate' : ''}</div>
        </div>}
        {results.metrics && signalIsLoss && <div className="stat stat--green">
          <span className="stat__label"><Activity size={11} /> Sample IoU / Dice</span>
          <div className="stat__value">{results.metrics.sample_iou} / {results.metrics.sample_dice}</div>
          <div className="stat__sub">Ground truth {Number(results.metrics.gt_defor_pixels).toLocaleString()} pixels</div>
        </div>}
        {results.detection_method === 'compare' && signalIsLoss && <>
          <div className="stat stat--cyan">
            <span className="stat__label">Temporal U-Net</span>
            <div className="stat__value">{results.temporal_deforestation_pct ?? 0}<span className="stat__unit">%</span></div>
            <div className="stat__sub">{Number(results.temporal_deforestation_hectares ?? 0).toLocaleString()} ha · dataset-labelled loss prediction</div>
          </div>
          <div className="stat stat--green">
            <span className="stat__label">Stage 1 baseline</span>
            <div className="stat__value">{results.baseline_deforestation_pct ?? 0}<span className="stat__unit">%</span></div>
            <div className="stat__sub">{Number(results.baseline_deforestation_hectares ?? 0).toLocaleString()} ha · T1 forest / T2 non-forest</div>
          </div>
        </>}
      </div>
      <div className="visual-grid fw-change-visuals">
        <VisualCard title="T1 · Before" src={results.t1_image || results.before_image} alt="Before image" />
        <VisualCard title="T2 · After" src={results.t2_image || results.after_image} alt="After image" />
        {results.before_forest_mask && <VisualCard title="T1 forest mask · Stage 1" src={results.before_forest_mask} alt="Forest mask before" />}
        {results.after_forest_mask && <VisualCard title="T2 forest mask · Stage 1" src={results.after_forest_mask} alt="Forest mask after" />}
        <VisualCard title="Forest change mask" src={results.forest_change_mask || results.deforestation_mask} alt="Deforestation mask" />
        <VisualCard title="Forest change overlay" src={results.forest_change_overlay || results.deforestation_overlay} alt="Deforestation overlay" />
        {signalIsLoss && results.temporal_deforestation_mask && <VisualCard title="Temporal U-Net prediction" src={results.temporal_deforestation_mask} alt="Temporal U-Net deforestation mask" />}
        {signalIsLoss && results.baseline_deforestation_mask && <VisualCard title="Stage 1 baseline" src={results.baseline_deforestation_mask} alt="Post-classification baseline mask" />}
      </div>
      {results.prediction_output?.invalid_pixel_encoding && <div className="note note--info">
        <Info size={13} /><span>Gray pixels in the selected prediction mask are invalid or cloud-masked; they are excluded from area totals.</span>
      </div>}
      {results.temporal_validation_iou != null && <div className="note note--info"><Info size={13} /><span>Checkpoint validation IoU: {Number(results.temporal_validation_iou).toFixed(3)} on Forest-Change validation data. This is a research metric, not a score for the live scene.</span></div>}
      {warnings.map((warning, index) => <div className="note note--warn" key={`${results.detection_method || 'unknown'}-${index}`}><Cpu size={13} /><span>{warning}</span></div>)}
    </section>
  );
}

function LatestForestResults({ results }) {
  return (
    <section className="fw-latest-results fw-latest-results--forest">
      <div className="fw-section-head">
        <div><h2>Latest forest analysis</h2><p>{results.sample_name || 'Single-image canopy segmentation'}</p></div>
        <span className="fw-results-badge">Forest U-Net · single image</span>
      </div>
      <div className="stat-grid fw-change-results">
        <div className="stat stat--green">
          <span className="stat__label"><Trees size={11} /> Forest canopy</span>
          <div className="stat__value">{results.forest_pct ?? '—'}<span className="stat__unit">%</span></div>
          <div className="stat__sub">{Number(results.forest_pixels ?? 0).toLocaleString()} pixels classified as canopy</div>
        </div>
        <div className="stat stat--neutral">
          <span className="stat__label">Non-forest</span>
          <div className="stat__value">{results.non_forest_pct ?? '—'}<span className="stat__unit">%</span></div>
          <div className="stat__sub">{Number(results.non_forest_pixels ?? 0).toLocaleString()} pixels</div>
        </div>
        <div className="stat stat--cyan">
          <span className="stat__label">{results.forest_hectares != null ? 'Estimated canopy area' : 'Image pixels'}</span>
          <div className="stat__value">
            {results.forest_hectares != null ? results.forest_hectares : Number(results.total_pixels ?? 0).toLocaleString()}
            <span className="stat__unit">{results.forest_hectares != null ? 'ha' : 'px'}</span>
          </div>
          <div className="stat__sub">{results.area_estimate_method || (results.forest_hectares == null ? 'Area unavailable without georeferencing' : 'Area estimate')}</div>
        </div>
      </div>
      <div className="visual-grid fw-forest-visuals">
        <VisualCard title="Original image" src={results.original_image} alt="Original satellite image" />
        <VisualCard title="Binary canopy mask" src={results.mask_image} alt="Predicted forest mask" />
        <VisualCard title="Canopy overlay" src={results.overlay_image} alt="Forest canopy overlay" />
      </div>
      {results.model_note && <div className="note note--warn"><Cpu size={13} /><span>{results.model_note}</span></div>}
    </section>
  );
}

function s2AreaEstimate(results) {
  return Boolean(results.area_estimate_method);
}

function formatChangeArea(area, unit) {
  const numericArea = Number(area);
  if (unit === 'ha') {
    return numericArea.toLocaleString(undefined, {
      minimumFractionDigits: 3,
      maximumFractionDigits: 3,
    });
  }
  return numericArea.toLocaleString(undefined, {
    minimumFractionDigits: 0,
    maximumFractionDigits: 4,
  });
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
  const signalPresentation = getForestChangeSignalPresentation(swipeData?.signal);
  const activeArea = swipeData?.deforestation_hectares ?? swipeData?.deforested_area ?? swipeData?.deforested_pixels ?? swipeData?.deforestation_pixels ?? 0;
  const areaUnit = swipeData?.area_unit === 'ha' || swipeData?.deforestation_hectares != null ? 'ha' : 'px';
  return (
    <>
      <section className="panel fw-history-selector">
        <div className="panel__head">
          <div>
            <div className="panel__title">Temporal Pair Comparison</div>
            <div className="block__hint">
              Scrub through the earlier image, the later image and the directional forest-change overlay.
            </div>
          </div>
        </div>

        <div className="control-grid fw-history-source">
          <div className="field">
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

          <button type="button" className="btn btn--primary" onClick={onLoad} disabled={loading || !swipeSample}>
            {loading ? <span className="btn__spinner" /> : <Play size={14} />}
            Load timeline
          </button>
        </div>
      </section>

      {swipeData ? (
        <section className="panel fw-timeline-result">
          <div className="panel__head">
            <div>
              <div className="panel__title">{selectedTimelineFrame?.label}</div>
              <div className="block__hint">{selectedTimelineFrame?.description}</div>
            </div>
            <div className="row" style={{ gap: 12 }}>
              <span className="badge badge--dim">{swipeData.pair_name}</span>
              {signalPresentation && <span className={`fw-results-badge fw-results-badge--${signalPresentation.tone}`}>
                {signalPresentation.label}
              </span>}
              {signalPresentation && <span className={`badge ${swipeData.signal === 'DEFORESTATION' ? 'badge--red' : 'badge--dim'}`}>
                {formatChangeArea(activeArea, areaUnit)} {areaUnit}
              </span>}
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
        <div className="empty-state fw-history-empty">
          <GitCompare size={22} />
          Select a temporal pair and load the timeline to compare T1, T2 and the predicted change.
        </div>
      )}
    </>
  );
}

function VisualCard({ title, src, alt }) {
  if (!src) return null;
  const filename = `${title.toLowerCase().replace(/[^a-z0-9]+/g, '-')}.png`;
  return (
    <div className="visual-card">
      <div className="visual-card__head"><span>{title}</span><a href={src} download={filename} aria-label={`Download ${title}`}>Download</a></div>
      <img className="visual-card__img" src={src} alt={alt} />
    </div>
  );
}
