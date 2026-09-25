/**
 * GisMap.jsx — Historical Satellite & Forest Change GIS Experience
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

// ── constants ─────────────────────────────────────────────────────────────────

const BASE_CENTER = [-3.4653, -62.2159];
const BASE_ZOOM = 13;

// ── component ─────────────────────────────────────────────────────────────────

export default function GisMap({ samples = [], aiMode, aiOpacity, onAiRun, aiOverlayData }) {
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

  // Sentinel search
  const [startDate, setStartDate] = useState('2023-01-01');
  const [endDate, setEndDate] = useState('2024-12-31');
  const [searching, setSearching] = useState(false);
  const [searchError, setSearchError] = useState('');
  const [acquisitions, setAcquisitions] = useState([]);  // [{id, datetime, cloud_coverage}]

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
  const analysisLayerRef = useRef(null); // L.imageOverlay for analysis overlay

  // AI overlay
  const [aiLayerVisible, setAiLayerVisible] = useState(true);


  // ── Map init ────────────────────────────────────────────────────────────────

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
    if (sentinelLayerRef.current) sentinelLayerRef.current.setOpacity(sentinelOpacity);
  }, [sentinelOpacity]);

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
      bbox:       `${west.toFixed(6)},${south.toFixed(6)},${east.toFixed(6)},${north.toFixed(6)}`,
      limit:      100,
    });

    try {
      const res = await fetch(`/api/sentinel/search?${params}`);
      const data = await res.json();
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
      setSearchError('Network error: ' + err.message);
    } finally {
      setSearching(false);
    }
  }, [aoiBounds, startDate, endDate]);


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

    const res  = await fetch(`/api/sentinel/preview?${params}`);
    const data = await res.json();

    if (!data.success || !data.image) {
      throw new Error(data.error || 'Failed to load Sentinel image.');
    }
    return data; // { image, bbox, width, height, ... }
  }, [aoiBounds]);

  /**
   * Select the FROM date (T1 = Before). Loads the image and shows it on the map.
   */
  const selectFrom = useCallback(async (idx) => {
    setFromIdx(idx);
    setFromData(null);
    setAnalysisResult(null);
    setAnalysisError('');
    setPreviewError('');
    setLoadingPreview(true);
    try {
      const data = await fetchPreview(acquisitions[idx]);
      if (!data) { setPreviewError('Failed to load FROM image.'); setLoadingPreview(false); return; }

      // Show FROM (before) on map
      const map = mapRef.current;
      const imgBounds = L.latLngBounds([data.bbox[1], data.bbox[0]], [data.bbox[3], data.bbox[2]]);
      if (beforeLayerRef.current) { map.removeLayer(beforeLayerRef.current); beforeLayerRef.current = null; }
      beforeLayerRef.current = L.imageOverlay(data.image, imgBounds, { opacity: sentinelOpacity, zIndex: 198 }).addTo(map);

      setFromData({ acq: acquisitions[idx], image: data.image, bbox: data.bbox });
      setLoadingPreview(false);
    } catch (err) {
      setPreviewError('FROM image: ' + err.message);
      setLoadingPreview(false);
    }
  }, [acquisitions, fetchPreview, sentinelOpacity]);

  /**
   * Select the TO date (T2 = After). Loads the image and shows it on the map.
   */
  const selectTo = useCallback(async (idx) => {
    setToIdx(idx);
    setToData(null);
    setAnalysisResult(null);
    setAnalysisError('');
    setPreviewError('');
    setLoadingPreview(true);
    try {
      const data = await fetchPreview(acquisitions[idx]);
      if (!data) { setPreviewError('Failed to load TO image.'); setLoadingPreview(false); return; }

      // Show TO (after) on map (on top of FROM)
      const map = mapRef.current;
      const imgBounds = L.latLngBounds([data.bbox[1], data.bbox[0]], [data.bbox[3], data.bbox[2]]);
      if (sentinelLayerRef.current) { map.removeLayer(sentinelLayerRef.current); sentinelLayerRef.current = null; }
      sentinelLayerRef.current = L.imageOverlay(data.image, imgBounds, { opacity: sentinelOpacity, zIndex: 200 }).addTo(map);
      map.fitBounds(imgBounds, { padding: [20, 20] });

      setToData({ acq: acquisitions[idx], image: data.image, bbox: data.bbox });
      setLoadingPreview(false);
    } catch (err) {
      setPreviewError('TO image: ' + err.message);
      setLoadingPreview(false);
    }
  }, [acquisitions, fetchPreview, sentinelOpacity]);

  // ── Deforestation Analysis ────────────────────────────────────────────────────

  const runAnalysis = useCallback(async () => {
    if (!fromData || !toData) {
      setAnalysisError('Select both a FROM date and a TO date before running analysis.');
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

    try {
      const res  = await fetch('/api/sentinel/analyze', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          image_before: fromData.image,
          image_after:  toData.image,
          date_before:  fromData.acq.datetime ? fromData.acq.datetime.slice(0, 10) : 'Unknown',
          date_after:   toData.acq.datetime   ? toData.acq.datetime.slice(0, 10)   : 'Unknown',
          bbox:         [west, south, east, north],
          threshold:    0.5,
        }),
      });
      const data = await res.json();

      if (!data.success) {
        setAnalysisError(data.error || 'Analysis failed.');
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
            opacity: 0.75, zIndex: 400,
          }).addTo(map);
        }
      }

      setAnalysisResult(data);
    } catch (err) {
      setAnalysisError('Network error: ' + err.message);
    } finally {
      setAnalysing(false);
    }
  }, [fromData, toData, aoiBounds]);

  // Toggle analysis overlay visibility
  const toggleAnalysisOverlay = useCallback(() => {
    if (!analysisLayerRef.current) return;
    if (analysisOverlayVisible) {
      analysisLayerRef.current.setOpacity(0);
    } else {
      analysisLayerRef.current.setOpacity(0.75);
    }
    setAnalysisOverlayVisible(v => !v);
  }, [analysisOverlayVisible]);

  // Fit map to AOI
  const fitToAoi = useCallback(() => {
    if (aoiBounds && mapRef.current) {
      mapRef.current.fitBounds(aoiBounds, { padding: [20, 20] });
    }
  }, [aoiBounds]);

  // ── Render ───────────────────────────────────────────────────────────────────

  const canAnalyze = fromData !== null && toData !== null &&
    fromIdx !== toIdx && !analysing && !loadingPreview;
  const bothSelected = fromIdx !== null && toIdx !== null;

  return (
    <div style={{ display: 'flex', height: '100%', gap: 16, minHeight: 0 }}>

      {/* ─── LEFT PANEL ─────────────────────────────────── */}
      <div style={{
        width: 320, display: 'flex', flexDirection: 'column', gap: 12,
        overflowY: 'auto', paddingRight: 4, flexShrink: 0,
      }}>

        {/* Base Map + AOI Tools */}
        <div className="glass-card" style={{ padding: 16 }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 12 }}>
            <Layers size={15} color="var(--primary-green)" />
            <span style={{ fontWeight: 700, fontSize: '0.9rem' }}>Map Controls</span>
          </div>

          {/* Base map switcher */}
          <div style={{ marginBottom: 12 }}>
            <div style={{ fontSize: '0.75rem', fontWeight: 600, color: 'var(--text-muted)', marginBottom: 6, textTransform: 'uppercase', letterSpacing: '0.5px' }}>Base Layer</div>
            <div className="toggle-group">
              <button className={`btn-toggle${baseMap === 'esri' ? ' active' : ''}`} onClick={() => setBaseMap('esri')}>
                Esri Satellite
              </button>
              <button className={`btn-toggle${baseMap === 'osm' ? ' active' : ''}`} onClick={() => setBaseMap('osm')}>
                Street Map
              </button>
            </div>
          </div>

          {/* AOI tools */}
          <div style={{ marginBottom: 4 }}>
            <div style={{ fontSize: '0.75rem', fontWeight: 600, color: 'var(--text-muted)', marginBottom: 6, textTransform: 'uppercase', letterSpacing: '0.5px' }}>Area of Interest (AOI)</div>
            <div style={{ display: 'flex', gap: 8 }}>
              {!isDrawing ? (
                <button
                  className="btn btn-primary"
                  style={{ flex: 1, padding: '9px 12px', fontSize: '0.82rem', background: '#f59e0b', boxShadow: '0 4px 12px rgba(245,158,11,0.3)' }}
                  onClick={startDrawing}
                >
                  <Square size={14} /> Draw AOI
                </button>
              ) : (
                <button
                  className="btn"
                  style={{ flex: 1, padding: '9px 12px', fontSize: '0.82rem', background: '#fee2e2', color: '#dc2626', border: '1px solid #fca5a5' }}
                  onClick={cancelDrawing}
                >
                  <X size={14} /> Cancel Draw
                </button>
              )}
              {aoiBounds && (
                <>
                  <button
                    className="btn"
                    style={{ padding: '9px 12px', fontSize: '0.82rem', background: 'var(--bg-main)', border: '1px solid var(--border-color)' }}
                    onClick={fitToAoi}
                    title="Fit map to AOI"
                  >
                    <MapPin size={14} />
                  </button>
                  <button
                    className="btn"
                    style={{ padding: '9px 12px', fontSize: '0.82rem', background: '#fee2e2', color: '#dc2626', border: '1px solid #fca5a5' }}
                    onClick={clearAoi}
                    title="Clear AOI"
                  >
                    <X size={14} />
                  </button>
                </>
              )}
            </div>
            {isDrawing && (
              <div style={{ marginTop: 8, padding: '6px 10px', background: '#fffbeb', border: '1px solid #fde68a', borderRadius: 6, fontSize: '0.78rem', color: '#92400e' }}>
                Click and drag on the map to draw your AOI rectangle.
              </div>
            )}
            {aoiBounds && !isDrawing && (() => {
              // Use the same bboxFromLatLngBounds that will be sent to the backend.
              const [w, s, e, n] = bboxFromLatLngBounds(aoiBounds);
              return (
                <div style={{ marginTop: 8, fontSize: '0.75rem', color: 'var(--text-muted)', lineHeight: 1.5 }}>
                  <strong>AOI:</strong> {s.toFixed(4)}°, {w.toFixed(4)}°<br />
                  to {n.toFixed(4)}°, {e.toFixed(4)}°
                </div>
              );
            })()}
          </div>
        </div>

        {/* Sentinel Search */}
        <div className="glass-card" style={{ padding: 16 }}>
          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 12 }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
              <Satellite size={15} color="#2563eb" />
              <span style={{ fontWeight: 700, fontSize: '0.9rem' }}>Sentinel-2 Search</span>
            </div>
            <span style={{ fontSize: '0.68rem', background: '#dbeafe', color: '#1d4ed8', padding: '2px 7px', borderRadius: 4, fontWeight: 700 }}>
              L2A
            </span>
          </div>

          {/* Date range */}
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 8, marginBottom: 12 }}>
            <div>
              <div style={{ fontSize: '0.72rem', fontWeight: 600, color: 'var(--text-muted)', marginBottom: 4 }}>Start Date</div>
              <input
                type="date"
                value={startDate}
                onChange={e => setStartDate(e.target.value)}
                style={{
                  width: '100%', padding: '8px 10px', borderRadius: 8, fontSize: '0.82rem',
                  border: '1px solid var(--border-color)', background: '#fff', color: 'var(--text-main)', outline: 'none',
                }}
              />
            </div>
            <div>
              <div style={{ fontSize: '0.72rem', fontWeight: 600, color: 'var(--text-muted)', marginBottom: 4 }}>End Date</div>
              <input
                type="date"
                value={endDate}
                onChange={e => setEndDate(e.target.value)}
                style={{
                  width: '100%', padding: '8px 10px', borderRadius: 8, fontSize: '0.82rem',
                  border: '1px solid var(--border-color)', background: '#fff', color: 'var(--text-main)', outline: 'none',
                }}
              />
            </div>
          </div>

          <button
            className="btn btn-primary full-width"
            style={{ padding: '10px 14px', fontSize: '0.85rem', background: '#2563eb', boxShadow: '0 4px 12px rgba(37,99,235,0.3)' }}
            onClick={runSearch}
            disabled={searching || !aoiBounds}
          >
            {searching
              ? <><Loader size={14} style={{ animation: 'spin 1s linear infinite' }} /> Searching…</>
              : <><Calendar size={14} /> Search Acquisitions</>
            }
          </button>

          {!aoiBounds && (
            <div style={{ marginTop: 8, fontSize: '0.75rem', color: 'var(--text-muted)', textAlign: 'center' }}>
              Draw an AOI first to search imagery.
            </div>
          )}

          {searchError && (
            <div style={{ marginTop: 8, padding: '8px 10px', background: '#fef2f2', border: '1px solid #fecaca', borderRadius: 6, fontSize: '0.78rem', color: '#b91c1c' }}>
              {searchError}
            </div>
          )}

          {acquisitions.length > 0 && (
            <div style={{ marginTop: 10, fontSize: '0.75rem', color: 'var(--text-muted)' }}>
              {acquisitions.length} acquisition{acquisitions.length !== 1 ? 's' : ''} found · click to load
            </div>
          )}
        </div>

        {/* Sentinel Layer Controls (only when loaded) */}
        {activeAcqMeta && (
          <div className="glass-card" style={{ padding: 16 }}>
            <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 10 }}>
              <span style={{ fontWeight: 700, fontSize: '0.88rem' }}>Active Sentinel Layer</span>
              <div style={{ display: 'flex', gap: 6 }}>
                <button
                  onClick={toggleSentinel}
                  style={{ background: 'none', border: '1px solid var(--border-color)', borderRadius: 6, padding: '4px 8px', cursor: 'pointer', display: 'flex', alignItems: 'center' }}
                  title={sentinelVisible ? 'Hide layer' : 'Show layer'}
                >
                  {sentinelVisible ? <Eye size={13} /> : <EyeOff size={13} />}
                </button>
                <button
                  onClick={clearSentinel}
                  style={{ background: '#fee2e2', border: '1px solid #fca5a5', borderRadius: 6, padding: '4px 8px', cursor: 'pointer', color: '#dc2626', display: 'flex', alignItems: 'center' }}
                  title="Remove Sentinel layer"
                >
                  <X size={13} />
                </button>
              </div>
            </div>

            <div style={{ fontSize: '0.78rem', color: 'var(--text-muted)', marginBottom: 8, lineHeight: 1.5 }}>
              <div><strong style={{ color: 'var(--text-main)' }}>{formatDate(activeAcqMeta.datetime)}</strong></div>
              <div style={{ display: 'flex', alignItems: 'center', gap: 4, marginTop: 2 }}>
                <Cloud size={11} color={cloudColor(activeAcqMeta.cloud_coverage)} />
                <span style={{ color: cloudColor(activeAcqMeta.cloud_coverage) }}>
                  {fmtCloud(activeAcqMeta.cloud_coverage)} cloud cover
                </span>
              </div>
            </div>

            {/* Opacity slider */}
            <div>
              <div className="slider-header">
                <span>Opacity</span>
                <span>{Math.round(sentinelOpacity * 100)}%</span>
              </div>
              <input
                type="range" min="0" max="1" step="0.05"
                value={sentinelOpacity}
                onChange={e => setSentinelOpacity(parseFloat(e.target.value))}
              />
            </div>

            {/* Before/After compare */}
            {beforeMeta && (
              <div style={{ marginTop: 12 }}>
                <div style={{ fontSize: '0.75rem', fontWeight: 600, color: 'var(--text-muted)', marginBottom: 6, textTransform: 'uppercase', letterSpacing: '0.5px' }}>Compare Mode</div>
                <div className="toggle-group">
                  <button className={`btn-toggle${compareMode === false ? ' active' : ''}`} onClick={() => activateCompare(false)}>Off</button>
                  <button className={`btn-toggle${compareMode === 'opacity' ? ' active' : ''}`} onClick={() => activateCompare('opacity')}>Opacity</button>
                </div>
              </div>
            )}
          </div>
        )}

        {/* AI Overlay Controls */}
        {aiOverlayData && (
          <div className="glass-card" style={{ padding: 16 }}>
            <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 8 }}>
              <span style={{ fontWeight: 700, fontSize: '0.88rem' }}>AI Overlay</span>
              <button
                onClick={() => setAiLayerVisible(v => !v)}
                style={{ background: 'none', border: '1px solid var(--border-color)', borderRadius: 6, padding: '4px 8px', cursor: 'pointer', display: 'flex', alignItems: 'center' }}
              >
                {aiLayerVisible ? <Eye size={13} /> : <EyeOff size={13} />}
              </button>
            </div>
            <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>
              {aiMode === 'stage1' ? '🌳 Forest canopy segmentation' : '🔴 Deforestation change detection'}
            </div>
            <div style={{ marginTop: 8, padding: '6px 8px', background: '#fffbeb', borderRadius: 6, fontSize: '0.72rem', color: '#92400e', display: 'flex', gap: 6, alignItems: 'flex-start' }}>
              <AlertTriangle size={11} style={{ marginTop: 1, flexShrink: 0 }} />
              AI results use training data samples. Live Sentinel imagery is NOT connected to the ML model.
            </div>
          </div>
        )}

        {/* Legend */}
        <div className="glass-card" style={{ padding: 14 }}>
          <div style={{ fontWeight: 700, fontSize: '0.82rem', marginBottom: 8 }}>Layer Legend</div>
          {[
            { color: '#f59e0b', label: 'AOI Boundary' },
            { color: '#3b82f6', label: 'Sentinel-2 RGB' },
            { color: '#059669', label: 'Stage 1: Forest Canopy' },
            { color: '#dc2626', label: 'Stage 2: Deforestation' },
          ].map(({ color, label }) => (
            <div key={label} style={{ display: 'flex', alignItems: 'center', gap: 8, marginTop: 5 }}>
              <div style={{ width: 12, height: 12, background: color, borderRadius: 3, flexShrink: 0 }} />
              <span style={{ fontSize: '0.77rem', color: 'var(--text-muted)' }}>{label}</span>
            </div>
          ))}
        </div>
      </div>

      {/* ─── MAP COLUMN ──────────────────────────────────── */}
      <div style={{ flex: 1, display: 'flex', flexDirection: 'column', gap: 0, minWidth: 0 }}>

        {/* Map */}
        <div style={{ flex: 1, position: 'relative', borderRadius: 16, overflow: 'hidden', border: '1px solid var(--border-color)', boxShadow: 'var(--shadow-md)' }}>
          <div ref={mapDivRef} style={{ width: '100%', height: '100%' }} />

          {/* Coordinate display */}
          {coords.lat && (
            <div style={{
              position: 'absolute', bottom: 10, left: '50%', transform: 'translateX(-50%)',
              background: 'rgba(15,23,42,0.75)', color: '#e2e8f0', borderRadius: 6,
              padding: '4px 10px', fontSize: '0.72rem', fontFamily: 'monospace',
              zIndex: 1000, backdropFilter: 'blur(4px)', pointerEvents: 'none',
            }}>
              {coords.lat}°, {coords.lon}°
            </div>
          )}

          {/* Loading spinner overlay */}
          {loadingPreview && (
            <div style={{
              position: 'absolute', inset: 0, background: 'rgba(255,255,255,0.55)',
              display: 'flex', alignItems: 'center', justifyContent: 'center',
              zIndex: 1500, backdropFilter: 'blur(2px)',
            }}>
              <div style={{ background: '#fff', borderRadius: 12, padding: '16px 24px', boxShadow: 'var(--shadow-lg)', display: 'flex', alignItems: 'center', gap: 12 }}>
                <Loader size={18} style={{ animation: 'spin 1s linear infinite', color: '#2563eb' }} />
                <span style={{ fontWeight: 600, fontSize: '0.88rem', color: 'var(--text-main)' }}>Loading Sentinel imagery…</span>
              </div>
            </div>
          )}

          {/* Preview error */}
          {previewError && (
            <div style={{
              position: 'absolute', top: 12, left: '50%', transform: 'translateX(-50%)',
              background: '#fef2f2', border: '1px solid #fca5a5', borderRadius: 8,
              padding: '8px 14px', fontSize: '0.8rem', color: '#b91c1c',
              zIndex: 1500, maxWidth: '80%', textAlign: 'center',
            }}>
              {previewError}
              <button onClick={() => setPreviewError('')} style={{ marginLeft: 8, background: 'none', border: 'none', cursor: 'pointer', color: '#b91c1c', fontWeight: 700 }}>✕</button>
            </div>
          )}
        </div>

        {/* ─── TIMELINE ──────────────────────────────────── */}
        {acquisitions.length > 0 && (
          <div style={{
            marginTop: 12,
            background: '#fff', border: '1px solid var(--border-color)',
            borderRadius: 14, padding: '14px 18px',
            boxShadow: 'var(--shadow-md)',
          }}>
            <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 10 }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                <Calendar size={14} color="#2563eb" />
                <span style={{ fontWeight: 700, fontSize: '0.88rem' }}>Sentinel-2 Historical Acquisitions</span>
                <span style={{ fontSize: '0.75rem', color: 'var(--text-muted)', background: 'var(--bg-main)', padding: '2px 8px', borderRadius: 10, border: '1px solid var(--border-color)' }}>
                  {acquisitions.length} available
                </span>
              </div>
              {beforeIdx !== null && selectedIdx !== null && beforeIdx !== selectedIdx && (
                <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
                  <span style={{ fontSize: '0.75rem', color: '#2563eb', fontWeight: 600 }}>
                    Compare: {formatDate(acquisitions[beforeIdx]?.datetime)} → {formatDate(acquisitions[selectedIdx]?.datetime)}
                  </span>
                  {!compareMode && beforeMeta && (
                    <button
                      onClick={() => activateCompare('opacity')}
                      style={{ fontSize: '0.72rem', padding: '3px 9px', borderRadius: 6, background: '#dbeafe', color: '#1d4ed8', border: '1px solid #bfdbfe', cursor: 'pointer', fontWeight: 600 }}
                    >
                      <GitCompare size={11} style={{ display: 'inline', marginRight: 3 }} />Compare
                    </button>
                  )}
                </div>
              )}
            </div>

            {/* Timeline scroll */}
            <div style={{ display: 'flex', gap: 6, overflowX: 'auto', paddingBottom: 4, scrollbarWidth: 'thin', alignItems: 'stretch' }}>
              {acquisitions.map((acq, idx) => {
                const isSelected = idx === selectedIdx;
                const isBefore   = idx === beforeIdx;
                const dateStr = formatDate(acq.datetime);
                const cc = acq.cloud_coverage;
                const ccColor = cloudColor(cc);

                return (
                  <div
                    key={acq.id || idx}
                    style={{
                      flexShrink: 0,
                      borderRadius: 10,
                      border: isSelected ? '2px solid #2563eb' : isBefore ? '2px solid #f59e0b' : '1px solid var(--border-color)',
                      background: isSelected ? '#2563eb' : isBefore ? '#fffbeb' : '#fff',
                      color: isSelected ? '#fff' : 'var(--text-main)',
                      cursor: 'pointer',
                      transition: 'all 0.15s',
                      minWidth: 96,
                      padding: '7px 10px',
                      display: 'flex', flexDirection: 'column', gap: 3,
                    }}
                    title={`ID: ${acq.id}`}
                  >
                    {/* Date */}
                    <div
                      style={{ fontSize: '0.78rem', fontWeight: 700, cursor: 'pointer' }}
                      onClick={() => selectAcquisition(idx)}
                    >
                      {dateStr}
                    </div>
                    {/* Cloud */}
                    <div style={{ display: 'flex', alignItems: 'center', gap: 3, fontSize: '0.7rem', color: isSelected ? 'rgba(255,255,255,0.8)' : ccColor }}>
                      <Cloud size={10} />
                      {fmtCloud(cc)}
                    </div>
                    {/* Before button */}
                    <button
                      onClick={(e) => { e.stopPropagation(); selectBeforeAcquisition(idx); }}
                      style={{
                        marginTop: 3, fontSize: '0.65rem', fontWeight: 600,
                        padding: '2px 6px', borderRadius: 5, cursor: 'pointer',
                        background: isBefore ? '#f59e0b' : (isSelected ? 'rgba(255,255,255,0.2)' : 'var(--bg-main)'),
                        color: isBefore ? '#fff' : (isSelected ? '#fff' : 'var(--text-muted)'),
                        border: isBefore ? '1px solid #f59e0b' : '1px solid var(--border-color)',
                      }}
                    >
                      {isBefore ? '★ Before' : 'Set Before'}
                    </button>
                  </div>
                );
              })}
            </div>

            {/* Compare mode instructions */}
            {beforeIdx !== null && selectedIdx !== null && beforeIdx !== selectedIdx && !compareMode && beforeMeta && (
              <div style={{ marginTop: 8, fontSize: '0.76rem', color: '#64748b' }}>
                Two acquisitions selected. Click <strong>Compare</strong> to overlay them with opacity blending.
              </div>
            )}
            {compareMode === 'opacity' && (
              <div style={{ marginTop: 8, padding: '6px 12px', background: '#eff6ff', borderRadius: 8, fontSize: '0.76rem', color: '#1d4ed8', display: 'flex', alignItems: 'center', gap: 8 }}>
                <Info size={12} />
                Opacity blend: <strong>{formatDate(acquisitions[beforeIdx]?.datetime)}</strong> (Before) and <strong>{formatDate(acquisitions[selectedIdx]?.datetime)}</strong> (After) — each at 50% opacity.
                <button onClick={() => activateCompare(false)} style={{ marginLeft: 'auto', background: 'none', border: '1px solid #bfdbfe', borderRadius: 5, padding: '2px 8px', cursor: 'pointer', color: '#1d4ed8', fontSize: '0.72rem' }}>Off</button>
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
}
