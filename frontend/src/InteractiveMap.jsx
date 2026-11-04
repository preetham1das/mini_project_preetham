/**
 * InteractiveMap.jsx — Historical Satellite & Forest Change GIS Experience
 *
 * Features:
 *  - Rectangle AOI drawing on the Leaflet map
 *  - Sentinel-2 STAC acquisition search over the drawn AOI
 *  - Historical date timeline with cloud coverage
 *  - Single-acquisition preview loaded as Leaflet imageOverlay
 *  - Before / After comparison (side-by-side + opacity swipe)
 *  - AI overlay layer (Stage 1 / Stage 2 results)
 *  - Base map switcher (Esri Satellite / OSM)
 *  - Coordinate display (mouse position)
 *  - Sentinel layer opacity slider
 *  - ML model DISCONNECTED from live Sentinel imagery (clearly labelled)
 */

import React, {
  useState,
  useEffect,
  useRef,
  useCallback,
  useMemo,
} from 'react';
import L from 'leaflet';
import {
  Satellite,
  Layers,
  MapPin,
  Calendar,
  Cloud,
  Eye,
  EyeOff,
  Square,
  X,
  Loader,
  GitCompare,
  Info,
  AlertTriangle,
  CheckCircle2,
  ArrowRight,
  Activity,
  Sparkles,
  Clock,
  Target,
  PanelLeftClose,
  PanelLeftOpen,
} from 'lucide-react';

// ── helpers ──────────────────────────────────────────────────────────────────

function formatDate(isoStr) {
  if (!isoStr) return '—';
  return isoStr.slice(0, 10);
}

function fmtCloud(v) {
  if (v === null || v === undefined) return '—';
  return `${Number(v).toFixed(1)}%`;
}

function cloudColor(v) {
  if (v === null || v === undefined) return '#64748b';
  if (v < 10) return '#059669';
  if (v < 30) return '#d97706';
  return '#dc2626';
}

/**
 * Normalize a longitude value to the range [-180, 180] using modular arithmetic.
 * This is the same calculation Leaflet uses internally in LatLng.wrap().
 * It does NOT clamp — it wraps, so e.g. 181 → -179, -435 → -75.
 */
function wrapLng(lng) {
  // Periodic wrap into [-180, 180)
  if (lng >= -180 && lng <= 180) return lng;
  return ((((lng + 180) % 360) + 360) % 360) - 180;
}

/**
 * Extract [west, south, east, north] from a Leaflet LatLngBounds.
 * Wraps longitudes to [-180, 180] to guard against Leaflet's continuous-world
 * coordinate system returning values outside that range when the map has
 * scrolled past the antimeridian or was initialized before layout settled.
 */
function bboxFromLatLngBounds(llBounds) {
  const west  = wrapLng(llBounds.getWest());
  const east  = wrapLng(llBounds.getEast());
  const south = llBounds.getSouth();
  const north = llBounds.getNorth();

  // Sanity-check: if wrapping inverted west/east (e.g. both wrapped to same side),
  // swap so west < east.
  return [
    west <= east ? west : east,
    south,
    west <= east ? east : west,
    north,
  ];
}

/**
 * Validate that a bbox [west, south, east, north] has geographic coordinates
 * within legal ranges. Returns an error string or null if valid.
 */
function validateBbox(west, south, east, north) {
  if (west < -180 || west > 180)  return `West longitude out of range: ${west}`;
  if (east < -180 || east > 180)  return `East longitude out of range: ${east}`;
  if (south < -90  || south > 90) return `South latitude out of range: ${south}`;
  if (north < -90  || north > 90) return `North latitude out of range: ${north}`;
  if (west >= east)               return `West (${west}) must be less than East (${east})`;
  if (south >= north)             return `South (${south}) must be less than North (${north})`;
  return null;
}

/**
 * Generate a transparent PNG data-URL highlighting cloud/obscured pixels in cyan.
 * Uses image luminance & saturation thresholding scaled to the known AOI cloud_pct.
 */
function generateCloudMaskDataUrl(imgSrc, cloudPct) {
  return new Promise((resolve) => {
    if (!imgSrc || !cloudPct || cloudPct <= 0) {
      resolve(null);
      return;
    }
    const img = new Image();
    img.crossOrigin = 'anonymous';
    img.onload = () => {
      try {
        const canvas = document.createElement('canvas');
        canvas.width = img.width;
        canvas.height = img.height;
        const ctx = canvas.getContext('2d');
        ctx.drawImage(img, 0, 0);
        const imgData = ctx.getImageData(0, 0, canvas.width, canvas.height);
        const data = imgData.data;
        const totalPixels = canvas.width * canvas.height;

        const scores = new Float32Array(totalPixels);
        for (let i = 0; i < totalPixels; i++) {
          const off = i * 4;
          const r = data[off];
          const g = data[off + 1];
          const b = data[off + 2];
          const lum = 0.299 * r + 0.587 * g + 0.114 * b;
          const maxDiff = Math.max(Math.abs(r - g), Math.abs(g - b), Math.abs(b - r));
          scores[i] = lum - maxDiff * 0.6;
        }

        const sampleSize = Math.min(totalPixels, 4096);
        const step = Math.max(1, Math.floor(totalPixels / sampleSize));
        const sampleScores = [];
        for (let i = 0; i < totalPixels; i += step) {
          sampleScores.push(scores[i]);
        }
        sampleScores.sort((a, b) => b - a);

        const cutoffIdx = Math.min(
          sampleScores.length - 1,
          Math.max(0, Math.floor((cloudPct / 100) * sampleScores.length))
        );
        const threshold = sampleScores[cutoffIdx];

        for (let i = 0; i < totalPixels; i++) {
          const off = i * 4;
          if (scores[i] >= threshold && scores[i] > 90) {
            // Cloud/obscured: bright cyan/sky-blue
            data[off]     = 56;   // R
            data[off + 1] = 189;  // G
            data[off + 2] = 248;  // B
            data[off + 3] = 210;  // Alpha
          } else {
            // Clear pixel: transparent
            data[off]     = 0;
            data[off + 1] = 0;
            data[off + 2] = 0;
            data[off + 3] = 0;
          }
        }

        ctx.putImageData(imgData, 0, 0);
        resolve(canvas.toDataURL('image/png'));
      } catch (e) {
        resolve(null);
      }
    };
    img.onerror = () => resolve(null);
    img.src = imgSrc;
  });
}

// ── constants ─────────────────────────────────────────────────────────────────

// Neutral world view (global overview) instead of hardcoded Amazon Basin
const BASE_CENTER = [20.0, 0.0];
const BASE_ZOOM = 2;

// ── component ─────────────────────────────────────────────────────────────────

export default function InteractiveMap({
  samples = [],
  aiMode,
  aiOpacity,
  onAiRun,
  aiOverlayData,
  stage2Threshold = 0.5,
  onStage2Result,
  onAcquisitions,
  workspace = 'map',
}) {
  // Map refs
  const mapDivRef   = useRef(null);
  const mapRef      = useRef(null);
  const esriLayerRef = useRef(null);
  const osmLayerRef  = useRef(null);
  const aoiLayerRef  = useRef(null);        // L.Rectangle for AOI
  const sentinelLayerRef = useRef(null);    // L.imageOverlay for active sentinel image
  const beforeLayerRef   = useRef(null);    // L.imageOverlay for "before" comparison
  const aiLayerRef       = useRef(null);    // L.imageOverlay for AI result
  const aiBoundsLayerRef = useRef(null);    // L.Rectangle for AI bounds

  // Drawing
  const drawingRef = useRef(false);
  const drawStartRef = useRef(null);
  const drawRectRef  = useRef(null);

  // UI state
  const [baseMap, setBaseMap] = useState('esri'); // 'esri' | 'osm'
  const [coords, setCoords] = useState({ lat: null, lon: null });
  const [aoiBounds, setAoiBounds] = useState(null);   // L.LatLngBounds | null
  const [isDrawing, setIsDrawing] = useState(false);
  const [drawerOpen, setDrawerOpen] = useState(true);  // floating control drawer

  // Sentinel search
  const [startDate, setStartDate] = useState('2023-01-01');
  const [endDate, setEndDate] = useState('2024-12-31');
  const [collection, setCollection] = useState('sentinel-2-l2a');
  const [maxCloudCover, setMaxCloudCover] = useState(100);
  const [searching, setSearching] = useState(false);
  const [searchError, setSearchError] = useState('');
  const [acquisitions, setAcquisitions] = useState([]);  // [{id, datetime, cloud_coverage}]
  const [selectedYear, setSelectedYear] = useState('all');
  const [hasSearched, setHasSearched] = useState(false);

  // Available unique years in acquisitions for filtering
  const availableYears = useMemo(() => {
    const years = new Set();
    acquisitions.forEach(a => {
      if (a.datetime) {
        years.add(a.datetime.substring(0, 4));
      }
    });
    return Array.from(years).sort();
  }, [acquisitions]);

  // Filtered acquisitions based on selected year tab
  const displayedAcquisitions = useMemo(() => {
    return acquisitions.filter((acquisition) =>
      (selectedYear === 'all' || acquisition.datetime?.startsWith(selectedYear)) &&
      (acquisition.cloud_coverage == null || acquisition.cloud_coverage <= maxCloudCover)
    );
  }, [acquisitions, selectedYear, maxCloudCover]);

  // Sentinel preview — Before/After workflow
  // fromIdx = FROM date (T1, older image = "Before")
  // toIdx   = TO date   (T2, newer image = "After")
  const [fromIdx, setFromIdx] = useState(null);
  const [toIdx,   setToIdx]   = useState(null);
  const [loadingPreview, setLoadingPreview] = useState(false);
  const [previewError,   setPreviewError]   = useState('');
  const [sentinelVisible, setSentinelVisible] = useState(true);
  const [sentinelOpacity, setSentinelOpacity] = useState(0.85);

  // Loaded image data for the two selected dates
  // { acq, image (base64), bbox, meta }
  const [fromData, setFromData] = useState(null);
  const [toData,   setToData]   = useState(null);

  // Deforestation analysis result
  const [analysing,       setAnalysing]       = useState(false);
  const [analysisError,   setAnalysisError]   = useState('');
  const [analysisResult,  setAnalysisResult]  = useState(null);
  const [analysisOverlayVisible, setAnalysisOverlayVisible] = useState(true);
  const [overlayOpacity, setOverlayOpacity] = useState(0.55);
  const analysisLayerRef = useRef(null); // L.imageOverlay for analysis overlay
  // Layer view mode on the Leaflet map: 'before' | 'after' | 'blend'
  const [mapViewMode, setMapViewMode] = useState('after');
  const [previewModalImg, setPreviewModalImg] = useState(null);

  // AI overlay
  const [aiLayerVisible, setAiLayerVisible] = useState(true);

  // SwinIR super-resolution (visualization only)
  const [srMode,     setSrMode]     = useState('original'); // 'original' | 'swinir'
  const [srLoading,  setSrLoading]  = useState(false);
  const [srData,     setSrData]     = useState(null);       // { image, width, height, note }
  const [srError,    setSrError]    = useState('');
  const [srAvailable, setSrAvailable] = useState(null);    // true | false | null (unknown)
  const srLayerRef = useRef(null);   // L.imageOverlay for SR image (replaces sentinel layer visually)

  // AOI-specific Cloud Statistics
  const [fromCloudStats, setFromCloudStats] = useState(null);
  const [toCloudStats,   setToCloudStats]   = useState(null);
  const [loadingFromCloud, setLoadingFromCloud] = useState(false);
  const [loadingToCloud,   setLoadingToCloud]   = useState(false);
  const [cloudMaskVisible, setCloudMaskVisible] = useState(false);
  const cloudLayerRef = useRef(null); // L.imageOverlay for cloud mask

  useEffect(() => {
    if (hasSearched) onAcquisitions?.(acquisitions);
  }, [acquisitions, hasSearched, onAcquisitions]);

  useEffect(() => {
    if (!mapDivRef.current || mapRef.current) return;

    const esri = L.tileLayer(
      'https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
      { maxZoom: 18, attribution: '© Esri' }
    );
    const osm = L.tileLayer(
      'https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png',
      { maxZoom: 19, attribution: '© OpenStreetMap' }
    );

    esriLayerRef.current = esri;
    osmLayerRef.current  = osm;

    const map = L.map(mapDivRef.current, {
      center: BASE_CENTER,
      zoom: BASE_ZOOM,
      layers: [esri],
      zoomControl: true,
      // KEY FIX: worldCopyJump prevents Leaflet from accumulating longitude
      // offsets that exceed ±180 when the user pans across the antimeridian.
      // Without this, e.latlng.lng can return values like 435° or -435°.
      worldCopyJump: true,
    });

    // Coordinate display uses .wrap() so the readout always shows [-180, 180].
    map.on('mousemove', (e) => {
      const lng = wrapLng(e.latlng.lng);
      setCoords({ lat: e.latlng.lat.toFixed(5), lon: lng.toFixed(5) });
    });
    map.on('mouseout', () => setCoords({ lat: null, lon: null }));

    mapRef.current = map;

    // Use ResizeObserver to call invalidateSize() whenever the container
    // dimensions actually change (replaces the unreliable fixed 200ms timeout).
    // This ensures the pixel-origin is correct before the user starts drawing.
    let ro = null;
    if (typeof ResizeObserver !== 'undefined') {
      ro = new ResizeObserver(() => {
        mapRef.current?.invalidateSize({ pan: false });
      });
      ro.observe(mapDivRef.current);
    } else {
      // Fallback for old browsers
      setTimeout(() => mapRef.current?.invalidateSize(), 300);
    }

    return () => {
      if (ro) ro.disconnect();
      map.remove();
      mapRef.current = null;
    };
  }, []);


  // ── Base map switch ──────────────────────────────────────────────────────────

  useEffect(() => {
    const map = mapRef.current;
    if (!map) return;
    if (baseMap === 'esri') {
      if (osmLayerRef.current)  map.removeLayer(osmLayerRef.current);
      if (esriLayerRef.current) esriLayerRef.current.addTo(map);
    } else {
      if (esriLayerRef.current) map.removeLayer(esriLayerRef.current);
      if (osmLayerRef.current)  osmLayerRef.current.addTo(map);
    }
  }, [baseMap]);

  // ── Opacity ──────────────────────────────────────────────────────────────────

  useEffect(() => {
    if (!sentinelVisible) {
      if (sentinelLayerRef.current) sentinelLayerRef.current.setOpacity(0);
      if (beforeLayerRef.current)   beforeLayerRef.current.setOpacity(0);
      if (srLayerRef.current)       srLayerRef.current.setOpacity(0);
      return;
    }
    if (srMode === 'swinir') {
      // SwinIR replaces the satellite display: show SwinIR layer with sentinelOpacity, hide originals
      if (srLayerRef.current)       srLayerRef.current.setOpacity(sentinelOpacity);
      if (sentinelLayerRef.current) sentinelLayerRef.current.setOpacity(0);
      if (beforeLayerRef.current)   beforeLayerRef.current.setOpacity(0);
      return;
    }
    // Normal mode: show original satellite layer(s), hide SwinIR
    if (srLayerRef.current) srLayerRef.current.setOpacity(0);
    if (mapViewMode === 'before') {
      if (beforeLayerRef.current)   beforeLayerRef.current.setOpacity(sentinelOpacity);
      if (sentinelLayerRef.current) sentinelLayerRef.current.setOpacity(0);
    } else if (mapViewMode === 'after') {
      if (sentinelLayerRef.current) sentinelLayerRef.current.setOpacity(sentinelOpacity);
      if (beforeLayerRef.current)   beforeLayerRef.current.setOpacity(0);
    } else if (mapViewMode === 'blend') {
      if (sentinelLayerRef.current) sentinelLayerRef.current.setOpacity(sentinelOpacity * 0.65);
      if (beforeLayerRef.current)   beforeLayerRef.current.setOpacity(sentinelOpacity * 0.65);
    }
  }, [mapViewMode, sentinelOpacity, sentinelVisible, srMode, fromData, toData]);

  useEffect(() => {
    if (analysisLayerRef.current) {
      analysisLayerRef.current.setOpacity(analysisOverlayVisible ? overlayOpacity : 0);
    }
  }, [overlayOpacity, analysisOverlayVisible]);

  useEffect(() => {
    if (aiLayerRef.current) aiLayerRef.current.setOpacity(aiLayerVisible ? (aiOpacity ?? 0.8) : 0);
  }, [aiLayerVisible, aiOpacity]);

  // ── AI overlay from parent ────────────────────────────────────────────────────

  useEffect(() => {
    const map = mapRef.current;
    if (!map) return;

    if (aiLayerRef.current)       { map.removeLayer(aiLayerRef.current); aiLayerRef.current = null; }
    if (aiBoundsLayerRef.current) { map.removeLayer(aiBoundsLayerRef.current); aiBoundsLayerRef.current = null; }

    if (!aiOverlayData) return;

    const { image, gis } = aiOverlayData;
    if (!image || !gis?.bounds) return;

    const bounds = L.latLngBounds(gis.bounds);
    aiBoundsLayerRef.current = L.rectangle(bounds, {
      color: aiMode === 'stage1' ? '#059669' : '#dc2626',
      weight: 2, fill: false, dashArray: '5 4',
    }).addTo(map);

    aiLayerRef.current = L.imageOverlay(image, bounds, {
      opacity: aiLayerVisible ? (aiOpacity ?? 0.8) : 0,
    }).addTo(map);

    map.fitBounds(bounds, { padding: [30, 30] });
  }, [aiOverlayData]); // eslint-disable-line react-hooks/exhaustive-deps

  // ── AOI Drawing ──────────────────────────────────────────────────────────────

  const startDrawing = useCallback(() => {
    const map = mapRef.current;
    if (!map) return;
    setIsDrawing(true);
    drawingRef.current = true;
    map.dragging.disable();
    map.getContainer().style.cursor = 'crosshair';
  }, []);

  const cancelDrawing = useCallback(() => {
    const map = mapRef.current;
    if (!map) return;
    setIsDrawing(false);
    drawingRef.current = false;
    map.dragging.enable();
    map.getContainer().style.cursor = '';
    if (drawRectRef.current) { map.removeLayer(drawRectRef.current); drawRectRef.current = null; }
    drawStartRef.current = null;
  }, []);

  const clearAoi = useCallback(() => {
    const map = mapRef.current;
    if (!map) return;
    if (aoiLayerRef.current) { map.removeLayer(aoiLayerRef.current); aoiLayerRef.current = null; }
    setAoiBounds(null);
    setAcquisitions([]);
    setFromIdx(null);
    setToIdx(null);
    setFromData(null);
    setToData(null);
    setAnalysisResult(null);
    setAnalysisError('');
    setSearchError('');
    setPreviewError('');
    if (sentinelLayerRef.current) { map.removeLayer(sentinelLayerRef.current); sentinelLayerRef.current = null; }
    if (beforeLayerRef.current)   { map.removeLayer(beforeLayerRef.current);   beforeLayerRef.current = null; }
    if (analysisLayerRef.current) { map.removeLayer(analysisLayerRef.current); analysisLayerRef.current = null; }
  }, []);

  useEffect(() => {
    const map = mapRef.current;
    if (!map) return;

    const onMouseDown = (e) => {
      if (!drawingRef.current) return;
      // .wrap() normalizes lng to [-180, 180] — prevents out-of-range coordinates
      // from Leaflet's continuous-world system reaching our AOI bounds.
      const wrappedLatLng = e.latlng.wrap();
      drawStartRef.current = wrappedLatLng;
      if (drawRectRef.current) { map.removeLayer(drawRectRef.current); drawRectRef.current = null; }
      drawRectRef.current = L.rectangle([wrappedLatLng, wrappedLatLng], {
        color: '#f59e0b', weight: 2, fillColor: '#fbbf24', fillOpacity: 0.15, dashArray: '5 4',
      }).addTo(map);
    };

    const onMouseMove = (e) => {
      if (!drawingRef.current || !drawStartRef.current || !drawRectRef.current) return;
      const wrappedLatLng = e.latlng.wrap();
      drawRectRef.current.setBounds(L.latLngBounds(drawStartRef.current, wrappedLatLng));
    };

    const onMouseUp = (e) => {
      if (!drawingRef.current || !drawStartRef.current) return;

      const wrappedEnd = e.latlng.wrap();
      const finalBounds = L.latLngBounds(drawStartRef.current, wrappedEnd);

      // Compute wrapped bbox for the size-check (consistent with what bboxFromLatLngBounds returns)
      const [west, south, east, north] = bboxFromLatLngBounds(finalBounds);

      // Log for debugging — these are the exact values that will be sent to the backend.
      console.debug('[AOI] Drawn bounds:', { west, south, east, north });

      // Validate before accepting the AOI
      const bboxErr = validateBbox(west, south, east, north);
      if (bboxErr) {
        console.error('[AOI] Invalid bbox after drawing:', bboxErr, { west, south, east, north });
        cancelDrawing();
        return;
      }

      // Too small? Ignore.
      if ((north - south) < 0.001 || (east - west) < 0.001) {
        cancelDrawing();
        return;
      }

      // Remove temp rect
      if (drawRectRef.current) { map.removeLayer(drawRectRef.current); drawRectRef.current = null; }

      // Remove old AOI
      if (aoiLayerRef.current) { map.removeLayer(aoiLayerRef.current); aoiLayerRef.current = null; }

      // Draw permanent AOI using the wrapped finalBounds
      aoiLayerRef.current = L.rectangle(finalBounds, {
        color: '#f59e0b', weight: 2.5, fillColor: '#fbbf24', fillOpacity: 0.08,
        dashArray: '6 4',
      }).addTo(map);

      setAoiBounds(finalBounds);
      setAcquisitions([]);
      setFromIdx(null);
      setToIdx(null);
      setFromData(null);
      setToData(null);
      setAnalysisResult(null);
      setAnalysisError('');
      setSearchError('');

      drawingRef.current = false;
      drawStartRef.current = null;
      setIsDrawing(false);
      map.dragging.enable();
      map.getContainer().style.cursor = '';
    };

    map.on('mousedown', onMouseDown);
    map.on('mousemove', onMouseMove);
    map.on('mouseup', onMouseUp);

    return () => {
      map.off('mousedown', onMouseDown);
      map.off('mousemove', onMouseMove);
      map.off('mouseup', onMouseUp);
    };
  }, [cancelDrawing]);


  const loadAmazonPreset = useCallback(() => {
    const map = mapRef.current;
    if (!map) return;
    const south = -3.515;
    const north = -3.415;
    const west = -62.265;
    const east = -62.165;
    const bounds = L.latLngBounds([[south, west], [north, east]]);

    if (drawRectRef.current) { map.removeLayer(drawRectRef.current); drawRectRef.current = null; }
    if (aoiLayerRef.current) { map.removeLayer(aoiLayerRef.current); aoiLayerRef.current = null; }

    aoiLayerRef.current = L.rectangle(bounds, {
      color: '#f59e0b', weight: 2.5, fillColor: '#fbbf24', fillOpacity: 0.08,
      dashArray: '6 4',
    }).addTo(map);

    setAoiBounds(bounds);
    setAcquisitions([]);
    setFromIdx(null);
    setToIdx(null);
    setFromData(null);
    setToData(null);
    setAnalysisResult(null);
    setAnalysisError('');
    setSearchError('');
    map.fitBounds(bounds, { padding: [20, 20] });
  }, []);

  // ── Sentinel Search ──────────────────────────────────────────────────────────

  const runSearch = useCallback(async () => {
    if (!aoiBounds) {
      setSearchError('Draw an Area of Interest on the map first.');
      return;
    }

    const [west, south, east, north] = bboxFromLatLngBounds(aoiBounds);

    // Pre-flight: validate bbox before sending to backend.
    const bboxErr = validateBbox(west, south, east, north);
    if (bboxErr) {
      const msg = `AOI coordinates are invalid: ${bboxErr}. Please redraw the AOI.`;
      console.error('[runSearch] Invalid bbox — aborting request:', { west, south, east, north }, bboxErr);
      setSearchError(msg);
      return;
    }

    const centerLat = (south + north) / 2;
    const centerLon = (west + east) / 2;

    // Log the exact values that will be sent to the backend.
    console.debug('[runSearch] bbox:', { west, south, east, north }, '| center:', { lat: centerLat, lon: centerLon });

    setSearching(true);
    setSearchError('');
    setAcquisitions([]);
    setSelectedYear('all');
    setHasSearched(false);
    setFromIdx(null);
    setToIdx(null);
    setFromData(null);
    setToData(null);
    setAnalysisResult(null);
    setAnalysisError('');

    const params = new URLSearchParams({
      lat:        centerLat.toFixed(6),
      lon:        centerLon.toFixed(6),
      start_date: startDate,
      end_date:   endDate,
      collection,
      bbox:       `${west.toFixed(6)},${south.toFixed(6)},${east.toFixed(6)},${north.toFixed(6)}`,
      limit:      1000,
    });

    try {
      const res = await fetch(`/api/sentinel/search?${params}`);
      const data = await res.json();
      setHasSearched(true);
      if (!data.success) {
        setSearchError(data.error || 'Search failed.');
      } else if (!data.acquisitions?.length) {
        setSearchError('No Sentinel-2 acquisitions found for this area and date range.');
      } else {
        // Sort oldest → newest for timeline
        const sorted = [...data.acquisitions].sort((a, b) =>
          (a.datetime || '').localeCompare(b.datetime || '')
        );
        setAcquisitions(sorted);
      }
    } catch (err) {
      setHasSearched(true);
      setSearchError('Network error: ' + err.message);
    } finally {
      setSearching(false);
    }
  }, [aoiBounds, startDate, endDate, collection]);


  // ── Load Sentinel Preview ─────────────────────────────────────────────────────

  /**
   * Fetch one Sentinel-2 acquisition image and return { image, bbox } data.
   * Does NOT place it on the map — caller decides what to do with the data.
   */
  const fetchPreview = useCallback(async (acq) => {
    const map = mapRef.current;
    if (!map || !acq) return null;

    const bounds = aoiBounds || map.getBounds();
    const [west, south, east, north] = bboxFromLatLngBounds(bounds);

    const bboxErr = validateBbox(west, south, east, north);
    if (bboxErr) {
      console.error('[fetchPreview] Invalid bbox:', bboxErr, { west, south, east, north });
      return null;
    }

    const centerLat = (south + north) / 2;
    const centerLon = (west + east) / 2;
    console.debug('[fetchPreview] bbox:', { west, south, east, north });

    // Size proportional to AOI aspect ratio
    const widthDeg  = east - west;
    const heightDeg = north - south;
    let imgW = 512, imgH = 512;
    if (widthDeg > 0 && heightDeg > 0) {
      const aspect = widthDeg / heightDeg;
      if (aspect >= 1) { imgW = 512; imgH = Math.round(512 / aspect); }
      else             { imgH = 512; imgW = Math.round(512 * aspect); }
      imgW = Math.min(Math.max(imgW, 128), 1024);
      imgH = Math.min(Math.max(imgH, 128), 1024);
    }

    const params = new URLSearchParams({
      acquisition_id: acq.id,
      lat: centerLat.toFixed(6),
      lon: centerLon.toFixed(6),
      bbox: `${west.toFixed(6)},${south.toFixed(6)},${east.toFixed(6)},${north.toFixed(6)}`,
      format: 'json',
      width:  imgW,
      height: imgH,
    });
    if (acq.datetime) params.append('datetime', acq.datetime);
    if (acq.collection) params.append('collection', acq.collection);

    const res  = await fetch(`/api/sentinel/preview?${params}`);
    const data = await res.json();

    if (!data.success || !data.image) {
      throw new Error(data.error || 'Failed to load Sentinel image.');
    }
    return data; // { image, bbox, width, height, ... }
  }, [aoiBounds]);

  /**
   * Fetch AOI-specific cloud statistics from /api/sentinel/cloud-stats.
   */
  const fetchCloudStats = useCallback(async (acq, bbox) => {
    if (!acq || !bbox) return null;
    const bboxStr = `${bbox[0].toFixed(6)},${bbox[1].toFixed(6)},${bbox[2].toFixed(6)},${bbox[3].toFixed(6)}`;
    const params = new URLSearchParams({
      acquisition_id: acq.id,
      bbox: bboxStr,
    });
    if (acq.datetime) params.append('datetime', acq.datetime);
    try {
      const res = await fetch(`/api/sentinel/cloud-stats?${params}`);
      const data = await res.json();
      if (data.success) return data;
      return null;
    } catch (err) {
      console.warn('Failed to fetch AOI cloud stats:', err);
      return null;
    }
  }, []);

  /**
   * Select the FROM date (T1 = Before). Loads the image and shows it on the map.
   * Accepts either an acquisition object or an index in acquisitions.
   */
  const selectFrom = useCallback(async (target) => {
    const acq = (typeof target === 'object' && target !== null) ? target : acquisitions[target];
    if (!acq) return;
    const realIdx = acquisitions.findIndex(a => a.id === acq.id);
    setFromIdx(realIdx >= 0 ? realIdx : 0);
    setFromData(null);
    setFromCloudStats(null);
    setAnalysisResult(null);
    setAnalysisError('');
    setPreviewError('');
    setLoadingPreview(true);
    try {
      const data = await fetchPreview(acq);
      if (!data) { setPreviewError('Failed to load FROM image.'); setLoadingPreview(false); return; }

      // Show FROM (before) on map
      const map = mapRef.current;
      const imgBounds = L.latLngBounds([data.bbox[1], data.bbox[0]], [data.bbox[3], data.bbox[2]]);
      if (beforeLayerRef.current) { map.removeLayer(beforeLayerRef.current); beforeLayerRef.current = null; }
      beforeLayerRef.current = L.imageOverlay(data.image, imgBounds, { opacity: sentinelOpacity, zIndex: 198 }).addTo(map);

      setFromData({ acq, image: data.image, bbox: data.bbox });
      setMapViewMode('before');
      setLoadingPreview(false);

      // Concurrently fetch AOI-specific cloud statistics
      setLoadingFromCloud(true);
      fetchCloudStats(acq, data.bbox)
        .then(stats => setFromCloudStats(stats))
        .catch(() => setFromCloudStats(null))
        .finally(() => setLoadingFromCloud(false));
    } catch (err) {
      setPreviewError('FROM image: ' + err.message);
      setLoadingPreview(false);
    }
  }, [acquisitions, fetchPreview, fetchCloudStats, sentinelOpacity]);

  /**
   * Select the TO date (T2 = After). Loads the image and shows it on the map.
   * Accepts either an acquisition object or an index in acquisitions.
   */
  const selectTo = useCallback(async (target) => {
    const acq = (typeof target === 'object' && target !== null) ? target : acquisitions[target];
    if (!acq) return;
    const realIdx = acquisitions.findIndex(a => a.id === acq.id);
    setToIdx(realIdx >= 0 ? realIdx : 0);
    setToData(null);
    setToCloudStats(null);
    setAnalysisResult(null);
    setAnalysisError('');
    setPreviewError('');
    setLoadingPreview(true);
    try {
      const data = await fetchPreview(acq);
      if (!data) { setPreviewError('Failed to load TO image.'); setLoadingPreview(false); return; }

      // Show TO (after) on map (on top of FROM)
      const map = mapRef.current;
      const imgBounds = L.latLngBounds([data.bbox[1], data.bbox[0]], [data.bbox[3], data.bbox[2]]);
      if (sentinelLayerRef.current) { map.removeLayer(sentinelLayerRef.current); sentinelLayerRef.current = null; }
      sentinelLayerRef.current = L.imageOverlay(data.image, imgBounds, { opacity: sentinelOpacity, zIndex: 200 }).addTo(map);
      map.fitBounds(imgBounds, { padding: [20, 20] });

      setToData({ acq, image: data.image, bbox: data.bbox });
      setMapViewMode('after');
      setLoadingPreview(false);

      // Concurrently fetch AOI-specific cloud statistics
      setLoadingToCloud(true);
      fetchCloudStats(acq, data.bbox)
        .then(stats => setToCloudStats(stats))
        .catch(() => setToCloudStats(null))
        .finally(() => setLoadingToCloud(false));
    } catch (err) {
      setPreviewError('TO image: ' + err.message);
      setLoadingPreview(false);
    }
  }, [acquisitions, fetchPreview, fetchCloudStats, sentinelOpacity]);

  // ── Deforestation Analysis ────────────────────────────────────────────────────

  const runAnalysis = useCallback(async () => {
    if (!fromData || !toData) {
      setAnalysisError('Select both a FROM date and a TO date before running analysis.');
      return;
    }
    if (!aoiBounds) {
      setAnalysisError('Draw an AOI and search for Sentinel-2 acquisitions before running analysis.');
      return;
    }
    if (new Date(fromData.acq.datetime) >= new Date(toData.acq.datetime)) {
      setAnalysisError('The FROM / T1 acquisition must be earlier than the TO / T2 acquisition.');
      return;
    }
    if (
      !Array.isArray(fromData.bbox) ||
      !Array.isArray(toData.bbox) ||
      fromData.bbox.length !== 4 ||
      toData.bbox.length !== 4 ||
      fromData.bbox.some((coordinate, index) =>
        !Number.isFinite(coordinate) ||
        !Number.isFinite(toData.bbox[index]) ||
        Math.abs(coordinate - toData.bbox[index]) > 0.01
      )
    ) {
      setAnalysisError('The Sentinel-2 images do not share the same area. Select both dates from the same AOI search.');
      return;
    }

    setAnalysing(true);
    setAnalysisError('');
    setAnalysisResult(null);

    // Clear previous analysis overlay
    const map = mapRef.current;
    if (analysisLayerRef.current && map) {
      map.removeLayer(analysisLayerRef.current);
      analysisLayerRef.current = null;
    }

    const [west, south, east, north] = bboxFromLatLngBounds(aoiBounds || map.getBounds());
    const bboxErr = validateBbox(west, south, east, north);
    if (bboxErr) {
      setAnalysisError('Invalid AOI: ' + bboxErr);
      setAnalysing(false);
      return;
    }

    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), 35000);

    try {
      const res = await fetch('/api/sentinel/analyze', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        signal: controller.signal,
        body: JSON.stringify({
          image_before: fromData.image,
          image_after:  toData.image,
          date_before:  fromData.acq.datetime ? fromData.acq.datetime.slice(0, 10) : 'Unknown',
          date_after:   toData.acq.datetime   ? toData.acq.datetime.slice(0, 10)   : 'Unknown',
          bbox:         [west, south, east, north],
          threshold:    stage2Threshold,
        }),
      });
      clearTimeout(timeoutId);

      let data;
      try {
        data = await res.json();
      } catch (jsonErr) {
        throw new Error(`Server returned HTTP ${res.status} with non-JSON response.`);
      }

      if (!res.ok || !data.success) {
        setAnalysisError(data.error || `Analysis failed (HTTP ${res.status}).`);
        setAnalysing(false);
        return;
      }

      // Place deforestation overlay on map using the TO date bbox
      if (data.deforestation_overlay && toData.bbox) {
        const imgBounds = L.latLngBounds(
          [toData.bbox[1], toData.bbox[0]],
          [toData.bbox[3], toData.bbox[2]]
        );
        if (map) {
          analysisLayerRef.current = L.imageOverlay(data.deforestation_overlay, imgBounds, {
            opacity: overlayOpacity, zIndex: 400,
          }).addTo(map);
        }
      }

      setAnalysisResult(data);
      onStage2Result?.(data);
    } catch (err) {
      clearTimeout(timeoutId);
      if (err.name === 'AbortError') {
        setAnalysisError('Analysis request timed out after 35 seconds. Please try again.');
      } else {
        setAnalysisError('Network error: ' + err.message);
      }
    } finally {
      clearTimeout(timeoutId);
      setAnalysing(false);
    }
  }, [fromData, toData, aoiBounds, overlayOpacity, stage2Threshold, onStage2Result]);

  // Toggle analysis overlay visibility
  const toggleAnalysisOverlay = useCallback(() => {
    setAnalysisOverlayVisible(v => !v);
  }, []);

  // Toggle sentinel layer visibility
  const toggleSentinel = useCallback(() => {
    setSentinelVisible(v => !v);
  }, []);

  // Clear Sentinel layers and reset selections
  const clearSentinel = useCallback(() => {
    const map = mapRef.current;
    if (!map) return;
    if (sentinelLayerRef.current) { map.removeLayer(sentinelLayerRef.current); sentinelLayerRef.current = null; }
    if (beforeLayerRef.current)   { map.removeLayer(beforeLayerRef.current);   beforeLayerRef.current = null; }
    if (analysisLayerRef.current) { map.removeLayer(analysisLayerRef.current); analysisLayerRef.current = null; }
    setFromIdx(null);
    setToIdx(null);
    setFromData(null);
    setToData(null);
    setAnalysisResult(null);
    setAnalysisError('');
    setPreviewError('');
    // Also reset SwinIR state
    if (srLayerRef.current) { map.removeLayer(srLayerRef.current); srLayerRef.current = null; }
    setSrMode('original');
    setSrData(null);
    setSrError('');
    // Also reset Cloud Mask state
    if (cloudLayerRef.current) { map.removeLayer(cloudLayerRef.current); cloudLayerRef.current = null; }
    setFromCloudStats(null);
    setToCloudStats(null);
    setCloudMaskVisible(false);
  }, []);

  // Check SwinIR checkpoint availability on mount
  useEffect(() => {
    fetch('/api/sentinel/superresolve/status')
      .then(r => r.json())
      .then(d => setSrAvailable(d.available === true))
      .catch(() => setSrAvailable(false));
  }, []);

  // Run SwinIR 4x super-resolution on the currently active Sentinel image
  const runSuperResolve = useCallback(async () => {
    // Determine which image to SR — matches the active map view (Before or After)
    const activeData = mapViewMode === 'before' ? fromData : (toData || fromData);
    if (!activeData) return;

    setSrLoading(true);
    setSrError('');
    setSrData(null);

    try {
      const res = await fetch('/api/sentinel/superresolve', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          image: activeData.image,
          bbox:  activeData.bbox,
        }),
      });
      const data = await res.json();

      if (!data.success) {
        setSrError(data.error || 'Super-resolution failed.');
        setSrMode('original');
        setSrLoading(false);
        return;
      }

      setSrData(data);
      setSrMode('swinir');

      // Replace the Leaflet satellite layer with the SR image
      // (same bbox, just 4x more pixels — visualization only)
      const map = mapRef.current;
      if (map && activeData.bbox) {
        const imgBounds = L.latLngBounds(
          [activeData.bbox[1], activeData.bbox[0]],
          [activeData.bbox[3], activeData.bbox[2]]
        );
        if (srLayerRef.current) { map.removeLayer(srLayerRef.current); srLayerRef.current = null; }
        srLayerRef.current = L.imageOverlay(data.image, imgBounds, {
          opacity: sentinelOpacity, zIndex: 201,
        }).addTo(map);

        // Hide original layers so they don't blend underneath the SR layer
        if (sentinelLayerRef.current) sentinelLayerRef.current.setOpacity(0);
        if (beforeLayerRef.current)   beforeLayerRef.current.setOpacity(0);
      }
    } catch (err) {
      setSrError('Network error: ' + err.message);
      setSrMode('original');
    } finally {
      setSrLoading(false);
    }
  }, [toData, fromData, mapViewMode, sentinelOpacity]);

  // Revert to original (remove SR overlay)
  const revertToOriginal = useCallback(() => {
    const map = mapRef.current;
    if (srLayerRef.current && map) { map.removeLayer(srLayerRef.current); srLayerRef.current = null; }
    setSrMode('original');
    setSrData(null);
    setSrError('');
    // Restore original satellite layer opacity
    if (mapViewMode === 'before') {
      if (beforeLayerRef.current)   beforeLayerRef.current.setOpacity(sentinelVisible ? sentinelOpacity : 0);
      if (sentinelLayerRef.current) sentinelLayerRef.current.setOpacity(0);
    } else if (mapViewMode === 'after') {
      if (sentinelLayerRef.current) sentinelLayerRef.current.setOpacity(sentinelVisible ? sentinelOpacity : 0);
      if (beforeLayerRef.current)   beforeLayerRef.current.setOpacity(0);
    } else if (mapViewMode === 'blend') {
      if (sentinelLayerRef.current) sentinelLayerRef.current.setOpacity(sentinelVisible ? sentinelOpacity * 0.65 : 0);
      if (beforeLayerRef.current)   beforeLayerRef.current.setOpacity(sentinelVisible ? sentinelOpacity * 0.65 : 0);
    }
  }, [mapViewMode, sentinelOpacity, sentinelVisible]);

  // Toggle Cloud Mask overlay on/off
  const toggleCloudMask = useCallback(() => {
    setCloudMaskVisible(v => !v);
  }, []);

  // Sync Cloud Mask overlay with map when active acquisition or cloudMaskVisible changes
  useEffect(() => {
    const map = mapRef.current;
    if (!map) return;

    if (!cloudMaskVisible) {
      if (cloudLayerRef.current) {
        map.removeLayer(cloudLayerRef.current);
        cloudLayerRef.current = null;
      }
      return;
    }

    const activeData = mapViewMode === 'before' ? fromData : (toData || fromData);
    const activeStats = mapViewMode === 'before' ? fromCloudStats : (toCloudStats || fromCloudStats);

    if (!activeData || !activeData.image || !activeData.bbox) {
      if (cloudLayerRef.current) {
        map.removeLayer(cloudLayerRef.current);
        cloudLayerRef.current = null;
      }
      return;
    }

    const cloudPct = activeStats?.cloud_pct ?? 0;
    generateCloudMaskDataUrl(activeData.image, cloudPct).then((maskUrl) => {
      if (!maskUrl) {
        if (cloudLayerRef.current) {
          map.removeLayer(cloudLayerRef.current);
          cloudLayerRef.current = null;
        }
        return;
      }
      const imgBounds = L.latLngBounds(
        [activeData.bbox[1], activeData.bbox[0]],
        [activeData.bbox[3], activeData.bbox[2]]
      );
      if (cloudLayerRef.current) {
        map.removeLayer(cloudLayerRef.current);
        cloudLayerRef.current = null;
      }
      cloudLayerRef.current = L.imageOverlay(maskUrl, imgBounds, {
        opacity: 0.75,
        zIndex: 250,
      }).addTo(map);
    });
  }, [cloudMaskVisible, mapViewMode, toData, fromData, toCloudStats, fromCloudStats]);

  // Fit map to AOI
  const fitToAoi = useCallback(() => {
    if (aoiBounds && mapRef.current) {
      mapRef.current.fitBounds(aoiBounds, { padding: [20, 20] });
    }
  }, [aoiBounds]);

  // ── Derived render values ─────────────────────────────────────────────────────

  const canAnalyze = fromData !== null && toData !== null &&
    fromData?.acq?.id !== toData?.acq?.id &&
    new Date(fromData.acq.datetime) < new Date(toData.acq.datetime) &&
    !analysing && !loadingPreview;

  const activeCloudStats = (mapViewMode === 'before' ? fromCloudStats : toCloudStats) || toCloudStats || fromCloudStats;

  // ── Render ───────────────────────────────────────────────────────────────────

  return (
    <div className={`map-stage ${drawerOpen ? '' : 'is-drawer-collapsed'}`}>

      {/* Leaflet canvas */}
      <div ref={mapDivRef} className="map-canvas" />

      {/* ─── Floating map toolbar ───────────────────────────────── */}
      <div className="map-toolbar">
        <div className="seg" role="group" aria-label="Base layer">
          <button
            type="button"
            className={`seg__btn ${baseMap === 'esri' ? 'is-active' : ''}`}
            onClick={() => setBaseMap('esri')}
            title="Esri World Imagery"
          >
            <Satellite size={12} /> Satellite
          </button>
          <button
            type="button"
            className={`seg__btn ${baseMap === 'osm' ? 'is-active' : ''}`}
            onClick={() => setBaseMap('osm')}
            title="OpenStreetMap street view"
          >
            <Layers size={12} /> Street
          </button>
        </div>

        <span className="divider-v" />

        {!isDrawing ? (
          <>
            <button type="button" className="btn btn--sm btn--ghost" onClick={startDrawing}>
              <Square size={12} /> Draw AOI
            </button>
            <button
              type="button"
              className="btn btn--sm btn--ghost"
              onClick={loadAmazonPreset}
              title="Amazon deforestation hotspot (-3.465°, -62.215°)"
            >
              <MapPin size={12} /> Amazon
            </button>
          </>
        ) : (
          <button type="button" className="btn btn--sm btn--danger" onClick={cancelDrawing}>
            <X size={12} /> Cancel
          </button>
        )}

        {aoiBounds && !isDrawing && (
          <>
            <button type="button" className="icon-btn" onClick={fitToAoi} title="Fit map to AOI">
              <Target size={14} />
            </button>
            <button type="button" className="icon-btn" onClick={clearAoi} title="Clear AOI and results">
              <X size={14} />
            </button>
          </>
        )}

        <span className="divider-v" />

        <button
          type="button"
          className={`icon-btn ${drawerOpen ? 'is-active' : ''}`}
          onClick={() => setDrawerOpen((open) => !open)}
          title={drawerOpen ? 'Hide control panel' : 'Show control panel'}
          aria-label={drawerOpen ? 'Hide control panel' : 'Show control panel'}
        >
          {drawerOpen ? <PanelLeftClose size={15} /> : <PanelLeftOpen size={15} />}
        </button>
      </div>

      {/* ─── Floating control drawer ────────────────────────────── */}
      <aside className={`drawer ${drawerOpen ? '' : 'is-collapsed'}`} aria-hidden={!drawerOpen}>
        <div className="drawer__head">
          <div className="drawer__title">
            <Satellite size={14} color="var(--cyan-bright)" /> {workspace === 'acquisition' ? 'Area & imagery' : 'Map controls'}
          </div>
          <button
            type="button"
            className="icon-btn"
            onClick={() => setDrawerOpen(false)}
            title="Hide panel"
          >
            <PanelLeftClose size={14} />
          </button>
        </div>

        <div className="drawer__body">

          {/* ── Area of Interest ─────────────────────────────── */}
          <div className="block">
            <div className="block__head">
              <div className="block__title">
                <MapPin size={14} color="var(--amber)" /> Area of Interest
              </div>
              {aoiBounds && <span className="badge badge--amber">AOI set</span>}
            </div>

            {isDrawing ? (
              <div className="note note--warn">
                <Info size={12} />
                <span>Click and drag on the map to draw your AOI rectangle.</span>
              </div>
            ) : (
              <div className="row" style={{ gap: 6 }}>
                <button type="button" className="btn btn--sm btn--ghost" style={{ flex: 1 }} onClick={startDrawing}>
                  <Square size={12} /> Draw
                </button>
                <button type="button" className="btn btn--sm btn--ghost" style={{ flex: 1 }} onClick={loadAmazonPreset}>
                  <MapPin size={12} /> Amazon
                </button>
              </div>
            )}

            {aoiBounds && !isDrawing && (() => {
              const [w, s, e, n] = bboxFromLatLngBounds(aoiBounds);
              return (
                <div className="block__hint" style={{ marginTop: 8, fontFamily: 'var(--font-mono)', fontSize: 10.5 }}>
                  {s.toFixed(4)}°, {w.toFixed(4)}° → {n.toFixed(4)}°, {e.toFixed(4)}°
                </div>
              );
            })()}

            {!aoiBounds && !isDrawing && (
              <div className="block__hint" style={{ marginTop: 7 }}>
                Draw a rectangle or load the Amazon preset to begin.
              </div>
            )}
          </div>

          {workspace === 'acquisition' && <>
          {/* ── Sentinel-2 search ────────────────────────────── */}
          <div className="block">
            <div className="block__head">
              <div className="block__title">
                <Satellite size={14} color="var(--cyan-bright)" /> Sentinel-2 Search
              </div>
              <span className="badge badge--cyan">{collection === 'all' ? 'L1C + L2A' : collection.endsWith('l1c') ? 'L1C' : 'L2A'}</span>
            </div>

            <div className="field" style={{ marginBottom: 9 }}>
              <label htmlFor="acq-collection">Sentinel-2 product</label>
              <select id="acq-collection" className="select" value={collection} onChange={(event) => setCollection(event.target.value)}>
                <option value="sentinel-2-l2a">Sentinel-2 L2A · Surface reflectance</option>
                <option value="sentinel-2-l1c">Sentinel-2 L1C · Top of atmosphere</option>
                <option value="all">L1C + L2A</option>
              </select>
            </div>

            <div className="field-grid">
              <div className="field">
                <label htmlFor="acq-start">Start date</label>
                <input
                  id="acq-start"
                  type="date"
                  className="input"
                  value={startDate}
                  onChange={(e) => setStartDate(e.target.value)}
                />
              </div>
              <div className="field">
                <label htmlFor="acq-end">End date</label>
                <input
                  id="acq-end"
                  type="date"
                  className="input"
                  value={endDate}
                  onChange={(e) => setEndDate(e.target.value)}
                />
              </div>
            </div>

            <button
              type="button"
              className="btn btn--primary btn--block btn--sm"
              style={{ marginTop: 9 }}
              onClick={runSearch}
              disabled={searching || !aoiBounds}
            >
              {searching ? (
                <>
                  <Loader size={13} style={{ animation: 'spin 1s linear infinite' }} /> Searching…
                </>
              ) : (
                <>
                  <Calendar size={13} /> Search acquisitions
                </>
              )}
            </button>

            {!aoiBounds && (
              <div className="block__hint" style={{ marginTop: 7, textAlign: 'center' }}>
                Draw an AOI first to search imagery.
              </div>
            )}

            {searchError && (
              <div className="note note--error" style={{ marginTop: 8 }}>
                <AlertTriangle size={11} />
                <span>{searchError}</span>
              </div>
            )}

            {acquisitions.length > 0 && (
              <div className="block__hint" style={{ marginTop: 8 }}>
                <strong style={{ color: 'var(--emerald-bright)' }}>{acquisitions.length}</strong> acquisition
                {acquisitions.length !== 1 ? 's' : ''} found — select FROM / TO on the timeline below.
              </div>
            )}
          </div>

          {/* ── Before / After workflow ──────────────────────── */}
          <div className="block">
            <div className="block__head">
              <div className="block__title">
                <GitCompare size={14} color="var(--emerald-bright)" /> Before / After
              </div>
              {(fromData || toData) && (
                <button type="button" className="btn btn--sm btn--danger" onClick={clearSentinel}>
                  <X size={11} /> Reset
                </button>
              )}
            </div>

            <p className="block__hint" style={{ marginBottom: 9 }}>
              Pick a <strong>FROM</strong> and a <strong>TO</strong> acquisition to feed the 9-channel Temporal U-Net.
            </p>

            <div className="tile-pair">
              {/* FROM / T1 */}
              <div className={`tile-card tile-card--t1 ${fromData ? 'is-set' : ''}`}>
                <div className="row row--between">
                  <span className="tile-card__tag">FROM / T1</span>
                  {fromData && (
                    <span
                      className="tile-card__cloud"
                      style={{ color: cloudColor(fromData.acq.cloud_coverage) }}
                      title="Scene-level tile cloud cover (100 × 100 km)"
                    >
                      ☁ {fmtCloud(fromData.acq.cloud_coverage)}
                    </span>
                  )}
                </div>

                <div className="tile-card__date">
                  {fromData ? formatDate(fromData.acq.datetime) : 'Choose FROM'}
                </div>

                {fromData ? (
                  <>
                    <img
                      className="tile-card__thumb"
                      src={fromData.image}
                      alt="FROM preview"
                      onClick={() =>
                        setPreviewModalImg({
                          title: `Before image (T1 · ${formatDate(fromData.acq.datetime)})`,
                          image: fromData.image,
                        })
                      }
                    />
                    <div className={`cloud-box ${fromCloudStats && fromCloudStats.cloud_pct > 20 ? 'is-bad' : ''}`}>
                      <div className="cloud-box__row" style={{ fontWeight: 700, marginBottom: 2 }}>
                        <span>Cloud in AOI</span>
                        {loadingFromCloud && <Loader size={8} style={{ animation: 'spin 1s linear infinite' }} />}
                      </div>
                      {fromCloudStats ? (
                        <>
                          <div className="cloud-box__row">
                            <span>Obscured <strong>{fromCloudStats.cloud_pct}%</strong></span>
                            <span style={{ color: 'var(--emerald-bright)' }}>
                              Usable <strong>{fromCloudStats.clear_pct}%</strong>
                            </span>
                          </div>
                          {fromCloudStats.cloud_pct > 20 && (
                            <div style={{ marginTop: 2, fontWeight: 600 }}>
                              <AlertTriangle size={9} style={{ verticalAlign: '-1px' }} /> High cloud cover
                            </div>
                          )}
                        </>
                      ) : (
                        <span>{loadingFromCloud ? 'Calculating…' : 'Unavailable'}</span>
                      )}
                    </div>
                  </>
                ) : (
                  <div className="block__hint">Click “Set FROM” in the timeline</div>
                )}
              </div>

              {/* TO / T2 */}
              <div className={`tile-card tile-card--t2 ${toData ? 'is-set' : ''}`}>
                <div className="row row--between">
                  <span className="tile-card__tag">TO / T2</span>
                  {toData && (
                    <span
                      className="tile-card__cloud"
                      style={{ color: cloudColor(toData.acq.cloud_coverage) }}
                      title="Scene-level tile cloud cover (100 × 100 km)"
                    >
                      ☁ {fmtCloud(toData.acq.cloud_coverage)}
                    </span>
                  )}
                </div>

                <div className="tile-card__date">
                  {toData ? formatDate(toData.acq.datetime) : 'Choose TO'}
                </div>

                {toData ? (
                  <>
                    <img
                      className="tile-card__thumb"
                      src={toData.image}
                      alt="TO preview"
                      onClick={() =>
                        setPreviewModalImg({
                          title: `After image (T2 · ${formatDate(toData.acq.datetime)})`,
                          image: toData.image,
                        })
                      }
                    />
                    <div className={`cloud-box ${toCloudStats && toCloudStats.cloud_pct > 20 ? 'is-bad' : ''}`}>
                      <div className="cloud-box__row" style={{ fontWeight: 700, marginBottom: 2 }}>
                        <span>Cloud in AOI</span>
                        {loadingToCloud && <Loader size={8} style={{ animation: 'spin 1s linear infinite' }} />}
                      </div>
                      {toCloudStats ? (
                        <>
                          <div className="cloud-box__row">
                            <span>Obscured <strong>{toCloudStats.cloud_pct}%</strong></span>
                            <span style={{ color: 'var(--emerald-bright)' }}>
                              Usable <strong>{toCloudStats.clear_pct}%</strong>
                            </span>
                          </div>
                          {toCloudStats.cloud_pct > 20 && (
                            <div style={{ marginTop: 2, fontWeight: 600 }}>
                              <AlertTriangle size={9} style={{ verticalAlign: '-1px' }} /> High cloud cover
                            </div>
                          )}
                        </>
                      ) : (
                        <span>{loadingToCloud ? 'Calculating…' : 'Unavailable'}</span>
                      )}
                    </div>
                  </>
                ) : (
                  <div className="block__hint">Click “Set TO” in the timeline</div>
                )}
              </div>
            </div>

            {/* Display + enhancement controls */}
            {(fromData || toData) && (
              <>
                <div className="divider" />

                {fromData && toData && (
                  <div className="seg" style={{ width: '100%', marginBottom: 9 }}>
                    <button
                      type="button"
                      className={`seg__btn ${mapViewMode === 'before' ? 'is-active' : ''}`}
                      style={{ flex: 1, justifyContent: 'center' }}
                      onClick={() => setMapViewMode('before')}
                    >
                      Before
                    </button>
                    <button
                      type="button"
                      className={`seg__btn ${mapViewMode === 'after' ? 'is-active' : ''}`}
                      style={{ flex: 1, justifyContent: 'center' }}
                      onClick={() => setMapViewMode('after')}
                    >
                      After
                    </button>
                    <button
                      type="button"
                      className={`seg__btn ${mapViewMode === 'blend' ? 'is-active' : ''}`}
                      style={{ flex: 1, justifyContent: 'center' }}
                      onClick={() => setMapViewMode('blend')}
                    >
                      Blend
                    </button>
                  </div>
                )}

                <div className="row row--between" style={{ marginBottom: 2 }}>
                  <span className="eyebrow" style={{ marginBottom: 0 }}>Satellite opacity</span>
                  <div className="row" style={{ gap: 6 }}>
                    <span className="slider-value">{Math.round(sentinelOpacity * 100)}%</span>
                    <button
                      type="button"
                      className="icon-btn"
                      style={{ width: 26, height: 26 }}
                      onClick={toggleSentinel}
                      title={sentinelVisible ? 'Hide satellite layer' : 'Show satellite layer'}
                    >
                      {sentinelVisible ? <Eye size={13} /> : <EyeOff size={13} />}
                    </button>
                  </div>
                </div>
                <input
                  type="range"
                  min="0"
                  max="1"
                  step="0.05"
                  value={sentinelOpacity}
                  onChange={(e) => setSentinelOpacity(parseFloat(e.target.value))}
                  aria-label="Satellite layer opacity"
                />

                {/* Cloud mask */}
                <div className="divider" />
                <div className="row row--between">
                  <div className="row" style={{ gap: 6 }}>
                    <Cloud size={13} color="var(--cyan-bright)" />
                    <span className="eyebrow" style={{ marginBottom: 0 }}>Cloud mask</span>
                  </div>
                  <button
                    type="button"
                    className={`btn btn--sm ${cloudMaskVisible ? 'btn--cyan' : 'btn--ghost'}`}
                    onClick={toggleCloudMask}
                  >
                    {cloudMaskVisible ? <Eye size={11} /> : <EyeOff size={11} />}
                    {cloudMaskVisible ? 'Active' : 'Off'}
                  </button>
                </div>
                {cloudMaskVisible && (
                  <div className="block__hint" style={{ marginTop: 5 }}>
                    <span
                      style={{
                        display: 'inline-block',
                        width: 8,
                        height: 8,
                        borderRadius: 2,
                        background: '#38BDF8',
                        marginRight: 5,
                        verticalAlign: 'middle',
                      }}
                    />
                    Cyan overlay marks {activeCloudStats ? `${activeCloudStats.cloud_pct}%` : 'the'} obscured pixels in the AOI.
                  </div>
                )}

                {/* SwinIR */}
                <div className="divider" />
                <div className="row row--between" style={{ marginBottom: 7 }}>
                  <div className="row" style={{ gap: 6 }}>
                    <Sparkles size={13} color="#C4B5FD" />
                    <span className="eyebrow" style={{ marginBottom: 0 }}>SwinIR 4×</span>
                  </div>
                  {srAvailable === false && <span className="badge badge--dim">No checkpoint</span>}
                </div>

                <div className="seg" style={{ width: '100%' }}>
                  <button
                    type="button"
                    className={`seg__btn ${srMode === 'original' ? 'is-active' : ''}`}
                    style={{ flex: 1, justifyContent: 'center' }}
                    onClick={revertToOriginal}
                    disabled={srLoading}
                  >
                    Original 10 m
                  </button>
                  <button
                    type="button"
                    className={`seg__btn ${srMode === 'swinir' ? 'is-active' : ''}`}
                    style={{ flex: 1, justifyContent: 'center' }}
                    onClick={srMode === 'swinir' ? undefined : runSuperResolve}
                    disabled={srAvailable === false || srLoading}
                    title={srAvailable === false ? 'SwinIR checkpoint not found' : 'Run SwinIR 4× super-resolution'}
                  >
                    {srLoading ? (
                      <>
                        <Loader size={11} style={{ animation: 'spin 1s linear infinite' }} /> Running…
                      </>
                    ) : (
                      'SwinIR 4×'
                    )}
                  </button>
                </div>

                {srMode === 'swinir' && srData && (
                  <div className="note note--info" style={{ marginTop: 7 }}>
                    <Sparkles size={11} />
                    <span>
                      4× visualisation · {srData.orig_width}×{srData.orig_height} → {srData.width}×{srData.height} px
                    </span>
                  </div>
                )}

                {srError && (
                  <div className="note note--error" style={{ marginTop: 7 }}>
                    <AlertTriangle size={11} />
                    <span>{srError}</span>
                  </div>
                )}

                <div className="block__hint" style={{ marginTop: 7, fontSize: 10 }}>
                  Super-resolution increases display pixel density; it does not create true 2.5 m measurements.
                </div>
              </>
            )}

            {/* Primary CTA */}
            <button
              type="button"
              className={`btn btn--primary btn--block btn--cta ${canAnalyze ? '' : 'btn--ghost'}`}
              style={{ marginTop: 11 }}
              disabled={!canAnalyze || analysing}
              onClick={runAnalysis}
            >
              {analysing ? (
                <>
                  <span className="btn__spinner" /> Analysing Forest Change…
                </>
              ) : (
                <>
                  <Activity size={16} /> Run Deforestation Analysis
                </>
              )}
            </button>

            {/* Validation hints */}
            {!aoiBounds && (
              <div className="block__hint" style={{ marginTop: 6, textAlign: 'center' }}>
                Draw an AOI on the map to begin.
              </div>
            )}
            {aoiBounds && (!fromData || !toData) && (
              <div className="block__hint" style={{ marginTop: 6, textAlign: 'center' }}>
                {!fromData && !toData && 'Select both FROM and TO dates from the timeline.'}
                {fromData && !toData && 'Now select a TO date from the timeline.'}
                {!fromData && toData && 'Now select a FROM date from the timeline.'}
              </div>
            )}
            {((fromData?.acq?.id && toData?.acq?.id && fromData.acq.id === toData.acq.id) ||
              (fromIdx !== null && toIdx !== null && fromIdx === toIdx)) && (
              <div className="note note--warn" style={{ marginTop: 7 }}>
                <AlertTriangle size={11} />
                <span>FROM and TO dates must be different.</span>
              </div>
            )}
            {fromData && toData &&
              fromData.acq.id !== toData.acq.id &&
              new Date(fromData.acq.datetime) > new Date(toData.acq.datetime) && (
                <div className="note note--warn" style={{ marginTop: 7 }}>
                  <AlertTriangle size={11} />
                  <span>FROM / T1 must be earlier than TO / T2.</span>
                </div>
              )}
            {analysisError && (
              <div className="note note--error" style={{ marginTop: 7 }}>
                <AlertTriangle size={11} />
                <span>{analysisError}</span>
              </div>
            )}
          </div>

          </>}

          {/* Legend */}
          <div className="block">
            <div className="block__head" style={{ marginBottom: 8 }}>
              <div className="block__title">
                <Layers size={13} color="var(--text-dim)" /> Layer legend
              </div>
            </div>
            <div className="legend" style={{ background: 'transparent', border: 'none', padding: 0, gap: 9 }}>
              {[
                { color: '#F59E0B', label: 'AOI boundary' },
                { color: '#3B82F6', label: 'Sentinel-2 RGB' },
                { color: '#EF4444', label: 'Forest loss' },
                { color: '#10B981', label: 'Canopy' },
              ].map(({ color, label }) => (
                <div key={label} className="legend__row">
                  <span className="legend__swatch" style={{ background: color }} />
                  <span style={{ fontSize: 10 }}>{label}</span>
                </div>
              ))}
            </div>
          </div>

        </div>
      </aside>

      {/* ─── Floating analytics panel ───────────────────────────── */}
      {workspace === 'acquisition' && analysisResult && (
        <aside className="analytics-panel">
          <div className="analytics-panel__head">
            <div className="analytics-panel__title">
              <CheckCircle2 size={14} color="var(--emerald-bright)" /> Deforestation Results
            </div>
            <span className="badge badge--green">Temporal U-Net</span>
          </div>

          <div className="analytics-panel__body">
            <div className="note note--ok">
              <Info size={11} />
              <span>
                Change between <strong>{analysisResult.date_before}</strong> and{' '}
                <strong>{analysisResult.date_after}</strong> across the selected AOI.
              </span>
            </div>

            <div className="stat-grid">
              <div className="stat stat--red">
                <span className="stat__label">
                  <Activity size={10} /> Loss
                </span>
                <div className="stat__value">
                  {analysisResult.deforestation_pct}
                  <span className="stat__unit">%</span>
                </div>
                <div className="stat__sub">canopy loss detected</div>
              </div>

              <div className="stat stat--red">
                <span className="stat__label">Area</span>
                <div className="stat__value">
                  {analysisResult.deforestation_hectares ?? Number(analysisResult.defor_pixels ?? 0).toLocaleString()}
                  <span className="stat__unit">{analysisResult.deforestation_hectares != null ? 'ha*' : 'px'}</span>
                </div>
                <div className="stat__sub">{analysisResult.area_estimate_method || 'Area unavailable'}</div>
              </div>

              <div className="stat stat--neutral">
                <span className="stat__label">Square km</span>
                <div className="stat__value">
                  {analysisResult.deforestation_sqkm ?? '—'}
                  {analysisResult.deforestation_sqkm != null && <span className="stat__unit">km²</span>}
                </div>
                <div className="stat__sub">AOI-based approximation</div>
              </div>

              <div className="stat stat--green">
                <span className="stat__label">
                  <Trees size={10} /> Stable
                </span>
                <div className="stat__value">
                  {analysisResult.stable_pct}
                  <span className="stat__unit">%</span>
                </div>
                <div className="stat__sub">preserved canopy</div>
              </div>
            </div>

            {/* Overlay controls */}
            <div>
              <div className="row row--between" style={{ marginBottom: 2 }}>
                <span className="eyebrow" style={{ marginBottom: 0 }}>Change overlay</span>
                <button
                  type="button"
                  className={`btn btn--sm ${analysisOverlayVisible ? 'btn--danger' : 'btn--ghost'}`}
                  onClick={toggleAnalysisOverlay}
                >
                  {analysisOverlayVisible ? <Eye size={11} /> : <EyeOff size={11} />}
                  {analysisOverlayVisible ? 'Visible' : 'Hidden'}
                </button>
              </div>
              <div className="slider-row">
                <span>Opacity</span>
                <span className="slider-value">{Math.round(overlayOpacity * 100)}%</span>
              </div>
              <input
                type="range"
                min="0.1"
                max="1.0"
                step="0.05"
                value={overlayOpacity}
                onChange={(e) => setOverlayOpacity(parseFloat(e.target.value))}
                aria-label="Change overlay opacity"
              />
            </div>

            <div className="legend">
              <div className="legend__row">
                <span className="legend__swatch" style={{ background: '#EF4444' }} />
                <span>Forest loss</span>
              </div>
              <div className="legend__row">
                <span className="legend__swatch" style={{ background: '#10B981' }} />
                <span>Stable forest</span>
              </div>
            </div>

            {/* Comparison previews */}
            <div>
              <div className="eyebrow">Comparison previews</div>
              <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 7 }}>
                {[
                  { label: `Before · ${analysisResult.date_before}`, image: analysisResult.before_image },
                  { label: `After · ${analysisResult.date_after}`, image: analysisResult.after_image },
                  { label: 'Change mask', image: analysisResult.deforestation_mask },
                  { label: 'Overlay', image: analysisResult.deforestation_overlay },
                ].map(({ label, image }) => (
                  <button
                    key={label}
                    type="button"
                    onClick={() => setPreviewModalImg({ title: label, image })}
                    style={{
                      padding: 5,
                      borderRadius: 8,
                      border: '1px solid var(--border)',
                      background: 'rgba(2,6,12,0.45)',
                      cursor: 'zoom-in',
                      textAlign: 'left',
                    }}
                  >
                    <img
                      src={image}
                      alt={label}
                      style={{ width: '100%', height: 58, objectFit: 'cover', borderRadius: 5, display: 'block' }}
                    />
                    <span style={{ display: 'block', marginTop: 4, fontSize: 9.5, color: 'var(--text-dim)' }}>
                      {label}
                    </span>
                  </button>
                ))}
              </div>
            </div>

            <div className="note note--warn">
              <AlertTriangle size={11} />
              <span>
                9-channel inference on live Sentinel-2 L2A bands (B04, B03, B02). Seasonal differences and cloud cover
                can affect the change signal.
              </span>
            </div>
          </div>
        </aside>
      )}

      {/* ─── Timeline dock ──────────────────────────────────────── */}
      {workspace === 'acquisition' && acquisitions.length > 0 && (
        <div className="timeline-dock">
          <div className="timeline-dock__head">
            <div className="row">
              <Calendar size={13} color="var(--cyan-bright)" />
              <span className="timeline-dock__title">Sentinel-2 Acquisitions</span>
              <span className="timeline-dock__count">
                {displayedAcquisitions.length} of {acquisitions.length}
              </span>
            </div>

            <div className="row row--wrap" style={{ gap: 6 }}>
              {fromData?.acq?.datetime ? (
                <span className="badge badge--amber">FROM · {formatDate(fromData.acq.datetime)}</span>
              ) : fromIdx !== null && acquisitions[fromIdx]?.datetime ? (
                <span className="badge badge--amber">FROM · {formatDate(acquisitions[fromIdx]?.datetime)}</span>
              ) : (
                <span className="badge badge--dim">FROM —</span>
              )}

              <ArrowRight size={11} color="var(--text-faint)" />

              {toData?.acq?.datetime ? (
                <span className="badge badge--cyan">TO · {formatDate(toData.acq.datetime)}</span>
              ) : toIdx !== null && acquisitions[toIdx]?.datetime ? (
                <span className="badge badge--cyan">TO · {formatDate(acquisitions[toIdx]?.datetime)}</span>
              ) : (
                <span className="badge badge--dim">TO —</span>
              )}

              {canAnalyze && (
                <span className="badge badge--green">
                  <CheckCircle2 size={10} /> Ready
                </span>
              )}
            </div>
          </div>

          {availableYears.length > 1 && (
            <div className="year-chips" style={{ marginBottom: 8 }}>
              <span style={{ fontSize: 10, color: 'var(--text-faint)', marginRight: 2 }}>Year</span>
              <button
                type="button"
                className={`chip ${selectedYear === 'all' ? 'is-active' : ''}`}
                onClick={() => setSelectedYear('all')}
              >
                All ({acquisitions.length})
              </button>
              {availableYears.map((yr) => {
                const yrCount = acquisitions.filter((a) => a.datetime && a.datetime.startsWith(yr)).length;
                return (
                  <button
                    key={yr}
                    type="button"
                    className={`chip ${selectedYear === yr ? 'is-active' : ''}`}
                    onClick={() => setSelectedYear(yr)}
                  >
                    {yr} ({yrCount})
                  </button>
                );
              })}
            </div>
          )}

          <div className="timeline-filter">
            <span>Maximum cloud cover</span><strong>{maxCloudCover}%</strong>
            <input type="range" min="0" max="100" step="5" value={maxCloudCover} onChange={(event) => setMaxCloudCover(Number(event.target.value))} aria-label="Maximum acquisition cloud cover" />
          </div>

          <div className="timeline-scroll">
            {displayedAcquisitions.map((acq, idx) => {
              const isFrom = fromData?.acq?.id
                ? fromData.acq.id === acq.id
                : fromIdx !== null && acquisitions[fromIdx]?.id === acq.id;
              const isTo = toData?.acq?.id
                ? toData.acq.id === acq.id
                : toIdx !== null && acquisitions[toIdx]?.id === acq.id;

              return (
                <div
                  key={acq.id || idx}
                  className={`acq-card ${isFrom ? 'is-from' : ''} ${isTo ? 'is-to' : ''}`}
                  title={`Acquisition: ${acq.id}\nDate: ${formatDate(acq.datetime)}\nCloud: ${fmtCloud(acq.cloud_coverage)}`}
                >
                  <div className="acq-card__date">{formatDate(acq.datetime)}</div>
                  <div className="acq-card__cloud" style={{ color: cloudColor(acq.cloud_coverage) }}>
                    <Cloud size={9} /> {fmtCloud(acq.cloud_coverage)}
                  </div>
                  <div className="acq-actions">
                    <button
                      type="button"
                      className={`acq-btn ${isFrom ? 'is-from' : ''}`}
                      onClick={() => selectFrom(acq)}
                      title="Set as Before (T1)"
                    >
                      {isFrom ? '★ FROM' : 'FROM'}
                    </button>
                    <button
                      type="button"
                      className={`acq-btn ${isTo ? 'is-to' : ''}`}
                      onClick={() => selectTo(acq)}
                      title="Set as After (T2)"
                    >
                      {isTo ? '★ TO' : 'TO'}
                    </button>
                  </div>
                </div>
              );
            })}
          </div>
        </div>
      )}

      {/* ─── Coordinate readout ─────────────────────────────────── */}
      {coords.lat && (
        <div className="coords" style={{ bottom: acquisitions.length > 0 ? 132 : 14 }}>
          {coords.lat}°, {coords.lon}°
        </div>
      )}

      {/* ─── Overlays: loading, errors, empty state ─────────────── */}
      {loadingPreview && (
        <div className="map-loading">
          <div className="map-loading__card">
            <Loader size={17} style={{ animation: 'spin 1s linear infinite', color: 'var(--cyan-bright)' }} />
            <span>Loading Sentinel imagery…</span>
          </div>
        </div>
      )}

      {previewError && (
        <div className="toast toast--error">
          <AlertTriangle size={13} />
          <span>{previewError}</span>
          <button type="button" className="toast__close" onClick={() => setPreviewError('')} aria-label="Dismiss">
            ✕
          </button>
        </div>
      )}

      {hasSearched && !searching && acquisitions.length === 0 && (
        <div
          className="empty-state glass"
          style={{
            position: 'absolute',
            top: '50%',
            left: '50%',
            transform: 'translate(-50%, -50%)',
            width: 'min(430px, calc(100% - 40px))',
            zIndex: 520,
          }}
        >
          <Calendar size={19} />
          <strong style={{ color: 'var(--text)', fontSize: 13 }}>No Sentinel-2 acquisitions found</strong>
          <span>
            Nothing was captured between {startDate} and {endDate} for this AOI. Sentinel-2 coverage starts mid-2015 —
            try widening the range or redrawing the AOI.
          </span>
        </div>
      )}

      {/* ─── Lightbox ───────────────────────────────────────────── */}
      {previewModalImg && (
        <div className="modal" onClick={() => setPreviewModalImg(null)}>
          <div className="modal__card" onClick={(e) => e.stopPropagation()}>
            <div className="modal__head">
              <span className="modal__title">{previewModalImg.title}</span>
              <button type="button" className="btn btn--sm btn--danger" onClick={() => setPreviewModalImg(null)}>
                <X size={11} /> Close
              </button>
            </div>
            <img className="modal__img" src={previewModalImg.image} alt={previewModalImg.title} />
          </div>
        </div>
      )}
    </div>
  );
}
