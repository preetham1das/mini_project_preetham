import React, { useState, useEffect, useCallback } from 'react';
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
} from 'lucide-react';
import InteractiveMap from './InteractiveMap.jsx';
import EvaluationPanel from './EvaluationPanel.jsx';

export default function App() {
  const [view, setView] = useState(() => window.location.hash.replace(/^#\/?/, '') || 'dashboard');

  const [loading, setLoading] = useState(false);
  const [backendStatus, setBackendStatus] = useState('connecting');
  const [sentinelStatus, setSentinelStatus] = useState('checking');

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
    setGisMode('stage2');
    setAiOverlayData({ image: data.deforestation_overlay, gis: data.gis, mode: 'stage2', metrics: data });
    recordAnalysis({
      type: 'Change detection',
      region: `Sentinel-2 ${data.date_before || 'T1'} → ${data.date_after || 'T2'}`,
      date: data.date_after || new Date().toISOString().slice(0, 10),
      lossHa: data.deforestation_hectares,
      lossPct: deforestationPct,
      lossPixels: deforestationPixels,
      areaMethod: data.area_estimate_method,
    });
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
        setAiOverlayData({ image: data.deforestation_overlay, gis: data.gis, mode: 'stage2', metrics: data });
        recordAnalysis({ type: 'Change detection', region: data.pair_name || 'Uploaded image pair', date: new Date().toISOString().slice(0, 10), lossHa: data.deforestation_hectares, lossPct: data.deforestation_pct, lossPixels: data.deforestation_pixels, areaMethod: data.area_estimate_method });
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

  /* Re-run Stage 2 inference on currently loaded live Sentinel-2 images */
  const handleRunLiveStage2 = (threshold = s2Threshold) => {
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

    setLoading(true);
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), 35000);

    fetch('/api/sentinel/analyze', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      signal: controller.signal,
      body: JSON.stringify({
        image_before: beforeImg,
        image_after: afterImg,
        date_before: dateBefore,
        date_after: dateAfter,
        bbox: bbox,
        threshold: threshold,
      }),
    })
      .then((res) => {
        clearTimeout(timeoutId);
        return res.json();
      })
      .then((data) => {
        setLoading(false);
        if (!data.success) {
          alert(data.error || 'Live analysis failed.');
          return;
        }
        handleLiveStage2Result(data);
      })
      .catch((err) => {
        clearTimeout(timeoutId);
        setLoading(false);
        if (err.name === 'AbortError') {
          alert('Live analysis timed out after 35 seconds. Please try again.');
        } else {
          alert('Failed to connect to backend: ' + err.message);
        }
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

  const navItems = [
    { id: 'dashboard', label: 'Dashboard', icon: LayoutDashboard },
    { id: 'map', label: 'GIS Map', icon: MapIcon },
    { id: 'acquisition', label: 'Satellite Acquisition', icon: Satellite },
    { id: 'forest', label: 'Forest Analysis', icon: Trees },
    { id: 'change', label: 'Change Detection', icon: GitCompare },
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
          <div className="fw-topbar__status"><span className={`fw-status-dot ${backendStatus}`} />{backendStatus === 'online' ? 'Services operational' : backendStatus === 'offline' ? 'API unavailable' : 'Checking services'}</div>
        </header>

        <main className={`app-main fw-content ${isMapPage ? 'fw-content--map' : ''}`}>
        {isMapPage ? (
          <div className="fw-map-page">
            <div className="fw-page-heading fw-page-heading--map">
              <div><span className="fw-eyebrow">{view === 'acquisition' ? 'IMAGERY WORKSPACE' : 'GEOSPATIAL WORKSPACE'}</span><h1>{view === 'acquisition' ? 'Satellite Acquisition' : 'Interactive GIS Map'}</h1><p>{view === 'acquisition' ? 'Define an area, search Sentinel-2 scenes, inspect previews and select a date pair for analysis.' : 'Explore satellite basemaps, define regions and review forest change overlays.'}</p></div>
              {view === 'acquisition' && <button className="btn btn--primary" onClick={() => navigate('change')}><GitCompare size={14} /> Open Change Detection</button>}
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
          <PageFrame className="fw-page--forest" title="Forest Analysis" description="Run Forest U-Net canopy segmentation on a sample tile or uploaded image.">
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
            <OverlaySettings
              overlay={aiOverlayData}
              opacity={gisOpacity}
              onOpacityChange={setGisOpacity}
              onViewMap={() => navigate('map')}
            />
          </PageFrame>
        ) : view === 'change' ? (
          <PageFrame className="fw-page--change" title="Change Detection" description="Compare aligned T1 and T2 imagery with the Temporal U-Net. Live Sentinel-2 pairs are selected on the map.">
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
            <OverlaySettings
              overlay={aiOverlayData}
              opacity={gisOpacity}
              onOpacityChange={setGisOpacity}
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
          <PageFrame title="Analytics & Reports" description="Explore recorded analysis results and export the data available in this browser.">
            <AnalyticsView history={analysisHistory} analytics={analytics} />
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
            {history.length ? <div className="fw-activity-list">{history.slice(0, 5).map((item) => <div className="fw-activity-row" key={item.id}><span className={`fw-activity-mark ${item.type === 'Forest analysis' ? 'forest' : 'loss'}`} /><div><strong>{item.type}</strong><small>{item.region || 'Analysis'} · {item.date || new Date(item.analyzedAt).toLocaleDateString()}</small></div><b>{item.lossPct != null ? `${item.lossPct}% loss` : item.forestPct != null ? `${item.forestPct}% canopy` : 'Completed'}</b></div>)}</div> : <EmptyState title="No completed analyses yet" text="Run a Forest U-Net or Temporal U-Net workflow to see results here." />}
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
        <input id="result-overlay-opacity" type="range" min="0.1" max="1" step="0.05" value={opacity} onChange={(event) => onOpacityChange(Number(event.target.value))} />
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

function AnalyticsView({ history, analytics }) {
  const lossRecords = history.filter((item) => item.lossPct != null && Number.isFinite(Number(item.lossPct)));
  const exportCsv = () => {
    const rows = [['Analysis date', 'Type', 'Region', 'Forest loss (ha estimate)', 'Forest loss (%)', 'Forest loss (pixels)', 'Forest cover (ha estimate)', 'Forest cover (%)', 'Forest cover (pixels)', 'Total pixels', 'Area estimate method'], ...history.map((item) => [item.date || '', item.type || '', item.region || '', item.areaMethod ? item.lossHa ?? '' : '', item.lossPct ?? '', item.lossPixels ?? '', item.areaMethod ? item.forestHa ?? '' : '', item.forestPct ?? '', item.forestPixels ?? '', item.totalPixels ?? '', item.areaMethod || ''])];
    const csv = rows.map((row) => row.map((cell) => `"${String(cell).replaceAll('"', '""')}"`).join(',')).join('\r\n');
    const url = URL.createObjectURL(new Blob([csv], { type: 'text/csv;charset=utf-8' }));
    const link = document.createElement('a'); link.href = url; link.download = 'forestwatch-analysis-history.csv'; link.click(); URL.revokeObjectURL(url);
  };
  const benchmarkSources = analytics?.stage1 && analytics?.stage2 ? '2 models' : analytics?.stage1 ? 'Forest U-Net' : analytics?.stage2 ? 'Temporal U-Net' : 'Unavailable';
  const forestRecords = history.filter((item) => Number.isFinite(Number(item.forestPct)));
  return <div className="fw-reports"><section className="fw-card"><div className="fw-section-head"><div><h2>Analysis report</h2><p>Export pixel metrics and any AOI-based area estimates recorded in this browser.</p></div><button className="btn btn--ghost" onClick={exportCsv} disabled={!history.length}>Download CSV</button></div><div className="fw-report-kpis"><div><span>Saved analyses</span><strong>{history.length}</strong></div><div><span>Change analyses</span><strong>{history.filter((item) => item.type === 'Change detection').length}</strong></div><div><span>Loss percentage records</span><strong>{lossRecords.length}</strong></div><div><span>Benchmark sources</span><strong>{benchmarkSources}</strong></div></div></section><section className="fw-card"><div className="fw-section-head"><div><h2>Forest loss by analysis</h2><p>Percentage of analyzed pixels classified as loss</p></div></div><TrendChart metric="lossPct" unit="%" records={lossRecords.slice(0, 12).reverse()} /></section><section className="fw-card"><div className="fw-section-head"><div><h2>Canopy cover by analysis</h2><p>Coverage reported by Forest U-Net runs</p></div></div><TrendChart metric="forestPct" unit="%" records={forestRecords.slice(0, 12).reverse()} /></section><section className="fw-card"><div className="fw-section-head"><div><h2>Analysis records</h2><p>Pixel counts and any AOI-based area estimates</p></div></div><HistoryTable history={history} /></section></div>;
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
              Run segmentation
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

      {s1Results && (
        <>
          <div className="stat-grid fw-forest-results" style={{ gridTemplateColumns: 'repeat(auto-fit, minmax(190px, 1fr))' }}>
            <div className="stat stat--green">
              <span className="stat__label">
                <Trees size={11} /> Forest canopy
              </span>
              <div className="stat__value">
                {s1Results.forest_pct}
                <span className="stat__unit">%</span>
              </div>
              <div className="stat__sub">{s1Results.forest_pixels.toLocaleString()} pixels classified as canopy</div>
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
              <span className="stat__label">Image pixels</span>
              <div className="stat__value">
                {s1Results.total_pixels.toLocaleString()}
                <span className="stat__unit">px</span>
              </div>
              <div className="stat__sub">Physical area unavailable without georeferencing</div>
            </div>
          </div>

          <div className="visual-grid fw-forest-visuals">
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
  onRunLive,
  onOpenMap,
}) {
  const [stage2Mode, setStage2Mode] = useState(
    s2Results?.live_source ? 'live' : (s2FileT1 || s2FileT2 ? 'upload' : s2Sample ? 'sample' : 'live')
  );

  return (
    <>
      <section className="panel fw-change-panel">
        <div className="panel__head">
          <div>
            <div className="panel__title">Bi-Temporal Deforestation Detection</div>
            <div className="block__hint">
              9-channel Temporal U-Net comparing earlier (T1) and later (T2) satellite imagery.
            </div>
          </div>
          <span className="badge badge--violet">
            <Layers size={11} /> Temporal U-Net · 9-ch
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
                    <strong>Active Live Pair:</strong> {s2Results.pair_name || 'Sentinel-2 Scene'}
                  </span>
                </div>

                <div className="slider-block">
                  <div className="slider-row">
                    <span>Deforestation confidence threshold</span>
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
                    Select an Area of Interest (AOI) rectangle on the GIS Map and choose two Sentinel-2 historical acquisition dates across 2018–2026. The 9-channel Temporal U-Net analyzes radiometric changes in the identical geographic tile.
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
                min="0.1"
                max="0.9"
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
                min="0.1"
                max="0.9"
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

          <div className="stat-grid fw-change-results" style={{ gridTemplateColumns: 'repeat(auto-fit, minmax(190px, 1fr))' }}>
            <div className="stat stat--red">
              <span className="stat__label">Forest loss</span>
              <div className="stat__value">
                {s2Results.deforestation_hectares != null ? s2Results.deforestation_hectares : Number(s2Results.deforestation_pixels ?? s2Results.defor_pixels ?? 0).toLocaleString()}
                <span className="stat__unit">{s2Results.deforestation_hectares != null ? 'ha*' : 'px'}</span>
              </div>
              <div className="stat__sub">{s2Results.deforestation_pct ?? 0}% of analyzed pixels classified as loss{s2Results.area_estimate_method ? ' · AOI estimate' : ''}</div>
            </div>
            <div className="stat stat--neutral">
              <span className="stat__label">Stable land</span>
              <div className="stat__value">
                {s2Results.stable_pct ?? 0}
                <span className="stat__unit">%</span>
              </div>
              <div className="stat__sub">{(s2Results.stable_pixels ?? 0).toLocaleString()} pixels</div>
            </div>
            {s2Results.metrics && (
              <div className="stat stat--green">
                <span className="stat__label">
                  <Activity size={11} /> Sample IoU / Dice
                </span>
                <div className="stat__value">
                  {s2Results.metrics.sample_iou} / {s2Results.metrics.sample_dice}
                </div>
                <div className="stat__sub">Ground truth {Number(s2Results.metrics.gt_defor_pixels).toLocaleString()} pixels</div>
              </div>
            )}
          </div>

          <div className="visual-grid fw-change-visuals" style={{ gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))' }}>
            <VisualCard title="T1 · Before" src={s2Results.t1_image || s2Results.before_image} alt="Before image" />
            <VisualCard title="T2 · After" src={s2Results.t2_image || s2Results.after_image} alt="After image" />
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
      <section className="panel fw-history-selector">
        <div className="panel__head">
          <div>
            <div className="panel__title">Temporal Pair Comparison</div>
            <div className="block__hint">
              Scrub through the earlier image, the later image and the predicted forest-loss overlay.
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
