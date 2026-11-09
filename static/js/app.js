document.addEventListener("DOMContentLoaded", () => {
    // Determine API Base URL
    const API_BASE = (window.location.protocol === "file:" || !window.location.origin || window.location.origin === "null")
        ? "http://127.0.0.1:5000"
        : "";

    // Nav & Tabs
    const navItems = document.querySelectorAll(".nav-item");
    const tabContents = document.querySelectorAll(".tab-content");
    const pageTitle = document.getElementById("page-title");
    const pageDesc = document.getElementById("page-desc");

    const tabInfo = {
        "gis-map": {
            title: "Interactive GIS Satellite Map Viewer",
            desc: "Geospatial analysis of Amazon basin canopy cover and deforestation detection using Leaflet.js and PyTorch."
        },
        stage1: {
            title: "Stage 1: Binary Forest Cover Segmentation",
            desc: "Extracting pixel-level forest canopy coverage from satellite imagery using Deep U-Net architecture."
        },
        stage2: {
            title: "Stage 2: Post-Classification Deforestation Detection",
            desc: "Segments T1 and T2 independently; loss is forest at T1 and non-forest at T2."
        },
        swipe: {
            title: "Temporal Swipe Comparison Viewer",
            desc: "Interactive split-screen swipe tool comparing Before (T1) and After (T2 + Deforestation Overlay)."
        },
        metrics: {
            title: "Model Performance Analytics & Validation Reports",
            desc: "Quantitative evaluation metrics (IoU, Dice, Precision, Recall) across dataset splits."
        }
    };

    let map = null;
    let currentOverlay = null;
    let currentBoundsLayer = null;
    let gisMode = "stage1";

    navItems.forEach(item => {
        item.addEventListener("click", () => {
            const targetTab = item.getAttribute("data-tab");
            
            navItems.forEach(n => n.classList.remove("active"));
            tabContents.forEach(c => c.classList.remove("active"));

            item.classList.add("active");
            document.getElementById(`tab-${targetTab}`).classList.add("active");

            if (tabInfo[targetTab]) {
                pageTitle.textContent = tabInfo[targetTab].title;
                pageDesc.textContent = tabInfo[targetTab].desc;
            }

            if (targetTab === "gis-map") {
                setTimeout(() => {
                    if (map) map.invalidateSize();
                    else initLeafletMap();
                }, 100);
            }
        });
    });

    // Initialize Leaflet Map
    function initLeafletMap() {
        if (map) return;

        const esriSatellite = L.tileLayer("https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}", {
            maxZoom: 18,
            attribution: "Tiles &copy; Esri &mdash; Source: Esri, i-cubed, USDA, USGS, AEX, GeoEye, Getmapping, Aerogrid, IGN, IGP, UPR-EGP, and the GIS User Community"
        });

        const osmStreet = L.tileLayer("https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png", {
            maxZoom: 19,
            attribution: "&copy; OpenStreetMap contributors"
        });

        map = L.map("leaflet-map", {
            center: [-3.4653, -62.2159],
            zoom: 13,
            layers: [esriSatellite]
        });

        const baseMaps = {
            "Satellite (Esri)": esriSatellite,
            "Street Map (OSM)": osmStreet
        };

        L.control.layers(baseMaps).addTo(map);
    }

    initLeafletMap();

    // Sliders & Controls
    const s1Threshold = document.getElementById("stage1-threshold");
    const s1ThresholdVal = document.getElementById("stage1-threshold-val");
    if (s1Threshold) {
        s1Threshold.addEventListener("input", (e) => {
            s1ThresholdVal.textContent = parseFloat(e.target.value).toFixed(2);
        });
    }

    const s2Threshold = document.getElementById("stage2-threshold");
    const s2ThresholdVal = document.getElementById("stage2-threshold-val");
    if (s2Threshold) {
        s2Threshold.addEventListener("input", (e) => {
            s2ThresholdVal.textContent = parseFloat(e.target.value).toFixed(2);
        });
    }

    const gisThreshold = document.getElementById("gis-threshold");
    const gisThresholdVal = document.getElementById("gis-threshold-val");
    if (gisThreshold) {
        gisThreshold.addEventListener("input", (e) => {
            gisThresholdVal.textContent = parseFloat(e.target.value).toFixed(2);
        });
    }

    const gisOpacity = document.getElementById("gis-opacity");
    const gisOpacityVal = document.getElementById("gis-opacity-val");
    if (gisOpacity) {
        gisOpacity.addEventListener("input", (e) => {
            const val = parseFloat(e.target.value);
            gisOpacityVal.textContent = `${Math.round(val * 100)}%`;
            if (currentOverlay) {
                currentOverlay.setOpacity(val);
            }
        });
    }

    // GIS Mode Toggle Buttons
    const btnGisS1 = document.getElementById("gis-mode-stage1");
    const btnGisS2 = document.getElementById("gis-mode-stage2");
    
    if (btnGisS1 && btnGisS2) {
        btnGisS1.addEventListener("click", () => {
            gisMode = "stage1";
            btnGisS1.classList.add("active");
            btnGisS2.classList.remove("active");
            populateGisDropdown();
        });

        btnGisS2.addEventListener("click", () => {
            gisMode = "stage2";
            btnGisS2.classList.add("active");
            btnGisS1.classList.remove("active");
            populateGisDropdown();
        });
    }

    const loadingOverlay = document.getElementById("loading-overlay");
    function showLoading() { if (loadingOverlay) loadingOverlay.style.display = "flex"; }
    function hideLoading() { if (loadingOverlay) loadingOverlay.style.display = "none"; }

    let samplesData = { stage1_samples: [], stage2_samples: [] };
    
    function populateGisDropdown() {
        const gisSelect = document.getElementById("gis-sample-select");
        if (!gisSelect) return;
        
        gisSelect.innerHTML = '<option value="">-- Choose Sample Satellite Tile --</option>';

        if (gisMode === "stage1") {
            samplesData.stage1_samples.forEach(sample => {
                const opt = document.createElement("option");
                opt.value = JSON.stringify(sample);
                opt.textContent = `[${sample.split}] ${sample.name}`;
                gisSelect.appendChild(opt);
            });
        } else {
            samplesData.stage2_samples.forEach(sample => {
                const opt = document.createElement("option");
                opt.value = JSON.stringify(sample);
                opt.textContent = sample.name;
                gisSelect.appendChild(opt);
            });
        }
    }

    // Fetch Sample Data
    fetch(`${API_BASE}/api/samples`)
        .then(res => {
            if (!res.ok) throw new Error(`HTTP Error ${res.status}`);
            return res.json();
        })
        .then(data => {
            samplesData = data;
            
            const s1Select = document.getElementById("stage1-sample-select");
            if (s1Select) {
                data.stage1_samples.forEach(sample => {
                    const opt = document.createElement("option");
                    opt.value = sample.path;
                    opt.textContent = `[${sample.split}] ${sample.name}`;
                    s1Select.appendChild(opt);
                });
            }

            const s2Select = document.getElementById("stage2-sample-select");
            const swipeSelect = document.getElementById("swipe-sample-select");

            data.stage2_samples.forEach(sample => {
                const opt = document.createElement("option");
                opt.value = JSON.stringify(sample);
                opt.textContent = sample.name;
                if (s2Select) s2Select.appendChild(opt);

                if (swipeSelect) {
                    const optSwipe = document.createElement("option");
                    optSwipe.value = JSON.stringify(sample);
                    optSwipe.textContent = sample.name;
                    swipeSelect.appendChild(optSwipe);
                }
            });

            populateGisDropdown();
        })
        .catch(err => {
            console.error("Backend connection error:", err);
            alert("⚠️ Backend Server Not Responding!\n\nPlease start the server in terminal:\ncd E:\\MINI_PROJECT\\Deforestation-Detection-from-Satellite-Images\n.\\venv\\Scripts\\python.exe app.py");
        });

    const s1FileInput = document.getElementById("stage1-file-input");
    const s1Filename = document.getElementById("stage1-filename");
    if (s1FileInput) {
        s1FileInput.addEventListener("change", (e) => {
            if (e.target.files.length > 0) {
                s1Filename.textContent = `Selected: ${e.target.files[0].name}`;
                const sel = document.getElementById("stage1-sample-select");
                if (sel) sel.value = "";
            }
        });
    }

    // Run GIS Map Spatial Analysis
    const gisRunBtn = document.getElementById("gis-run-btn");
    if (gisRunBtn) {
        gisRunBtn.addEventListener("click", () => {
            const gisSelectVal = document.getElementById("gis-sample-select").value;
            if (!gisSelectVal) {
                alert("Please select a sample satellite tile from the dropdown.");
                return;
            }

            const sampleObj = JSON.parse(gisSelectVal);
            const formData = new FormData();
            formData.append("threshold", gisThreshold.value);

            showLoading();

            if (gisMode === "stage1") {
                formData.append("sample_path", sampleObj.path);

                fetch(`${API_BASE}/api/predict/stage1`, { method: "POST", body: formData })
                    .then(res => res.json())
                    .then(data => {
                        hideLoading();
                        if (data.error) { alert(data.error); return; }

                        renderGisOverlay(data.overlay_image, data.gis.bounds, "Stage 1 Forest Canopy Cover");
                        
                        document.getElementById("gis-total-ha").textContent = `${data.total_hectares} ha (${data.forest_sqkm} km²)`;
                        document.getElementById("gis-stat-name").textContent = "Forest Canopy:";
                        document.getElementById("gis-stat-val").textContent = `${data.forest_hectares} ha (${data.forest_pct}%)`;
                        document.getElementById("gis-bounds-txt").textContent = `${data.gis.center[0]}, ${data.gis.center[1]}`;
                        document.getElementById("gis-metrics-card").style.display = "flex";
                    })
                    .catch(err => {
                        hideLoading();
                        console.error(err);
                        alert("Failed to fetch prediction from backend server. Is app.py running?");
                    });
            } else {
                formData.append("sample_id", sampleObj.id);
                formData.append("split", sampleObj.split);

                fetch(`${API_BASE}/api/predict/stage2`, { method: "POST", body: formData })
                    .then(res => res.json())
                    .then(data => {
                        hideLoading();
                        if (data.error) { alert(data.error); return; }

                        renderGisOverlay(data.deforestation_overlay, data.gis.bounds, "Stage 2 Deforestation Mask");
                        
                        document.getElementById("gis-total-ha").textContent = "655.36 ha (6.55 km²)";
                        document.getElementById("gis-stat-name").textContent = "Deforestation Loss:";
                        document.getElementById("gis-stat-val").textContent = `${data.deforestation_hectares} ha (${data.deforestation_pct}%)`;
                        document.getElementById("gis-bounds-txt").textContent = `${data.gis.center[0]}, ${data.gis.center[1]}`;
                        document.getElementById("gis-metrics-card").style.display = "flex";
                    })
                    .catch(err => {
                        hideLoading();
                        console.error(err);
                        alert("Failed to fetch prediction from backend server. Is app.py running?");
                    });
            }
        });
    }

    function renderGisOverlay(imgBase64, bounds, title) {
        initLeafletMap();

        if (currentOverlay) map.removeLayer(currentOverlay);
        if (currentBoundsLayer) map.removeLayer(currentBoundsLayer);

        const leafletBounds = L.latLngBounds(bounds);

        currentBoundsLayer = L.rectangle(leafletBounds, {
            color: "#10b981",
            weight: 2,
            fill: false,
            dashArray: "4, 4"
        }).addTo(map);

        const opacity = parseFloat(document.getElementById("gis-opacity").value);
        currentOverlay = L.imageOverlay(imgBase64, leafletBounds, { opacity: opacity }).addTo(map);

        map.fitBounds(leafletBounds, { padding: [30, 30] });
    }

    // Run Stage 1 Segmentation
    const s1RunBtn = document.getElementById("stage1-run-btn");
    if (s1RunBtn) {
        s1RunBtn.addEventListener("click", () => {
            const formData = new FormData();
            formData.append("threshold", s1Threshold.value);

            const file = s1FileInput ? s1FileInput.files[0] : null;
            const samplePath = document.getElementById("stage1-sample-select").value;

            if (file) {
                formData.append("file", file);
            } else if (samplePath) {
                formData.append("sample_path", samplePath);
            } else {
                alert("Please select a sample image or upload a custom image file.");
                return;
            }

            showLoading();

            fetch(`${API_BASE}/api/predict/stage1`, { method: "POST", body: formData })
                .then(res => res.json())
                .then(data => {
                    hideLoading();
                    if (data.error) { alert(data.error); return; }

                    document.getElementById("s1-stat-ha").textContent = `${data.forest_hectares} ha`;
                    document.getElementById("s1-stat-pct").textContent = `${data.forest_pct}% (${data.forest_pixels.toLocaleString()} px)`;
                    document.getElementById("s1-stat-nonpct").textContent = `${data.non_forest_pct}%`;
                    document.getElementById("s1-stat-nonpx").textContent = `${data.non_forest_pixels.toLocaleString()} px`;
                    document.getElementById("s1-stat-total-ha").textContent = `${data.total_hectares} ha`;

                    document.getElementById("stage1-stats").style.display = "grid";

                    document.getElementById("s1-img-orig").src = data.original_image;
                    document.getElementById("s1-img-mask").src = data.mask_image;
                    document.getElementById("s1-img-overlay").src = data.overlay_image;

                    document.getElementById("stage1-visuals").style.display = "grid";
                })
                .catch(err => {
                    hideLoading();
                    console.error(err);
                    alert("Failed to fetch prediction from backend server. Is app.py running?");
                });
        });
    }

    // Run Stage 2 post-classification forest-loss detection
    const s2RunBtn = document.getElementById("stage2-run-btn");
    if (s2RunBtn) {
        s2RunBtn.addEventListener("click", () => {
            const s2SelectVal = document.getElementById("stage2-sample-select").value;
            if (!s2SelectVal) {
                alert("Please select a temporal pair from the dropdown.");
                return;
            }

            const sampleObj = JSON.parse(s2SelectVal);
            const formData = new FormData();
            formData.append("sample_id", sampleObj.id);
            formData.append("split", sampleObj.split);
            formData.append("threshold", s2Threshold.value);

            showLoading();

            fetch(`${API_BASE}/api/predict/stage2`, { method: "POST", body: formData })
                .then(res => res.json())
                .then(data => {
                    hideLoading();
                    if (data.error) { alert(data.error); return; }

                    document.getElementById("s2-stat-defor-ha").textContent = `${data.deforestation_hectares} ha`;
                    document.getElementById("s2-stat-defor-pct").textContent = `${data.deforestation_pct}% (${data.deforestation_pixels.toLocaleString()} px)`;
                    document.getElementById("s2-stat-stable-pct").textContent = `${data.stable_pct}%`;
                    document.getElementById("s2-stat-stable-px").textContent = `${data.stable_pixels.toLocaleString()} px`;

                    if (data.metrics) {
                        document.getElementById("s2-stat-metrics").textContent = `${data.metrics.sample_iou} / ${data.metrics.sample_dice}`;
                        document.getElementById("s2-stat-gt-ha").textContent = `Ground Truth: ${data.metrics.gt_defor_hectares} ha (${data.metrics.gt_defor_pct}%)`;
                        document.getElementById("s2-card-gt").style.display = "flex";
                    } else {
                        document.getElementById("s2-card-gt").style.display = "none";
                    }

                    document.getElementById("stage2-stats").style.display = "grid";

                    document.getElementById("s2-img-t1").src = data.t1_image;
                    document.getElementById("s2-img-t2").src = data.t2_image;
                    document.getElementById("s2-img-mask").src = data.deforestation_mask;
                    document.getElementById("s2-img-overlay").src = data.deforestation_overlay;

                    document.getElementById("stage2-visuals").style.display = "grid";
                })
                .catch(err => {
                    hideLoading();
                    console.error(err);
                    alert("Failed to fetch prediction from backend server. Is app.py running?");
                });
        });
    }

    // Swipe Viewer Integration
    const swipeLoadBtn = document.getElementById("swipe-load-btn");
    if (swipeLoadBtn) {
        swipeLoadBtn.addEventListener("click", () => {
            const swipeVal = document.getElementById("swipe-sample-select").value;
            if (!swipeVal) {
                alert("Please select a temporal pair for the swipe viewer.");
                return;
            }

            const sampleObj = JSON.parse(swipeVal);
            const formData = new FormData();
            formData.append("sample_id", sampleObj.id);
            formData.append("split", sampleObj.split);
            formData.append("threshold", 0.5);

            showLoading();

            fetch(`${API_BASE}/api/predict/stage2`, { method: "POST", body: formData })
                .then(res => res.json())
                .then(data => {
                    hideLoading();
                    if (data.error) { alert(data.error); return; }

                    document.getElementById("swipe-img-before").src = data.t1_image;
                    document.getElementById("swipe-img-after").src = data.deforestation_overlay;

                    document.getElementById("swipe-wrapper").style.display = "flex";
                    initSwipeSlider();
                })
                .catch(err => {
                    hideLoading();
                    console.error(err);
                    alert("Failed to fetch prediction from backend server. Is app.py running?");
                });
        });
    }

    // Swipe Drag Logic
    function initSwipeSlider() {
        const box = document.getElementById("swipe-box");
        const beforeWrapper = document.getElementById("swipe-before-wrapper");
        const handle = document.getElementById("swipe-handle");

        if (!box || !beforeWrapper || !handle) return;

        let isDragging = false;

        function setPosition(x) {
            const rect = box.getBoundingClientRect();
            let posX = x - rect.left;
            if (posX < 0) posX = 0;
            if (posX > rect.width) posX = rect.width;

            const percentage = (posX / rect.width) * 100;
            beforeWrapper.style.width = `${percentage}%`;
            handle.style.left = `${percentage}%`;
        }

        handle.addEventListener("mousedown", () => isDragging = true);
        window.addEventListener("mouseup", () => isDragging = false);
        window.addEventListener("mousemove", (e) => {
            if (isDragging) setPosition(e.clientX);
        });

        handle.addEventListener("touchstart", () => isDragging = true);
        window.addEventListener("touchend", () => isDragging = false);
        window.addEventListener("touchmove", (e) => {
            if (isDragging && e.touches.length > 0) setPosition(e.touches[0].clientX);
        });
    }
});
