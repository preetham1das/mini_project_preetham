document.addEventListener("DOMContentLoaded", () => {
    // Tab Navigation
    const navItems = document.querySelectorAll(".nav-item");
    const tabContents = document.querySelectorAll(".tab-content");
    const pageTitle = document.getElementById("page-title");
    const pageDesc = document.getElementById("page-desc");

    const tabInfo = {
        stage1: {
            title: "Stage 1: Binary Forest Cover Segmentation",
            desc: "Extracting pixel-level forest canopy coverage from satellite imagery using Deep U-Net architecture."
        },
        stage2: {
            title: "Stage 2: Temporal Deforestation Change Detection",
            desc: "Detecting forest loss (deforestation) between T1 and T2 satellite image pairs using 6-Channel Temporal U-Net."
        },
        metrics: {
            title: "Model Performance Analytics & Validation Reports",
            desc: "Quantitative evaluation metrics (IoU, Dice, Precision, Recall) across dataset splits."
        }
    };

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
        });
    });

    // Sliders
    const s1Threshold = document.getElementById("stage1-threshold");
    const s1ThresholdVal = document.getElementById("stage1-threshold-val");
    s1Threshold.addEventListener("input", (e) => {
        s1ThresholdVal.textContent = parseFloat(e.target.value).toFixed(2);
    });

    const s2Threshold = document.getElementById("stage2-threshold");
    const s2ThresholdVal = document.getElementById("stage2-threshold-val");
    s2Threshold.addEventListener("input", (e) => {
        s2ThresholdVal.textContent = parseFloat(e.target.value).toFixed(2);
    });

    // Loading overlay helper
    const loadingOverlay = document.getElementById("loading-overlay");
    function showLoading() { loadingOverlay.style.display = "flex"; }
    function hideLoading() { loadingOverlay.style.display = "none"; }

    // Fetch sample lists
    let samplesData = { stage1_samples: [], stage2_samples: [] };
    
    fetch("/api/samples")
        .then(res => res.json())
        .then(data => {
            samplesData = data;
            
            // Populate Stage 1 samples dropdown
            const s1Select = document.getElementById("stage1-sample-select");
            data.stage1_samples.forEach(sample => {
                const opt = document.createElement("option");
                opt.value = sample.path;
                opt.textContent = `[${sample.split}] ${sample.name}`;
                s1Select.appendChild(opt);
            });

            // Populate Stage 2 samples dropdown
            const s2Select = document.getElementById("stage2-sample-select");
            data.stage2_samples.forEach(sample => {
                const opt = document.createElement("option");
                opt.value = JSON.stringify(sample);
                opt.textContent = sample.name;
                s2Select.appendChild(opt);
            });
        })
        .catch(err => console.error("Failed to load sample options:", err));

    // File Dropzone filename display
    const s1FileInput = document.getElementById("stage1-file-input");
    const s1Filename = document.getElementById("stage1-filename");
    s1FileInput.addEventListener("change", (e) => {
        if (e.target.files.length > 0) {
            s1Filename.textContent = `Selected: ${e.target.files[0].name}`;
            document.getElementById("stage1-sample-select").value = "";
        }
    });

    // Run Stage 1 Segmentation
    document.getElementById("stage1-run-btn").addEventListener("click", () => {
        const formData = new FormData();
        formData.append("threshold", s1Threshold.value);

        const file = s1FileInput.files[0];
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

        fetch("/api/predict/stage1", {
            method: "POST",
            body: formData
        })
        .then(res => res.json())
        .then(data => {
            hideLoading();
            if (data.error) {
                alert(data.error);
                return;
            }

            // Display Statistics
            document.getElementById("s1-stat-pct").textContent = `${data.forest_pct}%`;
            document.getElementById("s1-stat-px").textContent = `${data.forest_pixels.toLocaleString()} pixels`;
            document.getElementById("s1-stat-nonpct").textContent = `${data.non_forest_pct}%`;
            document.getElementById("s1-stat-nonpx").textContent = `${data.non_forest_pixels.toLocaleString()} pixels`;
            document.getElementById("s1-stat-total").textContent = data.total_pixels.toLocaleString();

            document.getElementById("stage1-stats").style.display = "grid";

            // Display Images
            document.getElementById("s1-img-orig").src = data.original_image;
            document.getElementById("s1-img-mask").src = data.mask_image;
            document.getElementById("s1-img-overlay").src = data.overlay_image;

            document.getElementById("stage1-visuals").style.display = "grid";
        })
        .catch(err => {
            hideLoading();
            console.error(err);
            alert("Error running Stage 1 inference.");
        });
    });

    // Run Stage 2 Temporal Change Detection
    document.getElementById("stage2-run-btn").addEventListener("click", () => {
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

        fetch("/api/predict/stage2", {
            method: "POST",
            body: formData
        })
        .then(res => res.json())
        .then(data => {
            hideLoading();
            if (data.error) {
                alert(data.error);
                return;
            }

            // Display Stats
            document.getElementById("s2-stat-defor-pct").textContent = `${data.deforestation_pct}%`;
            document.getElementById("s2-stat-defor-px").textContent = `${data.deforestation_pixels.toLocaleString()} pixels`;
            document.getElementById("s2-stat-stable-pct").textContent = `${data.stable_pct}%`;
            document.getElementById("s2-stat-stable-px").textContent = `${data.stable_pixels.toLocaleString()} pixels`;

            if (data.metrics) {
                document.getElementById("s2-stat-metrics").textContent = `${data.metrics.sample_iou} / ${data.metrics.sample_dice}`;
                document.getElementById("s2-card-gt").style.display = "flex";
            } else {
                document.getElementById("s2-card-gt").style.display = "none";
            }

            document.getElementById("stage2-stats").style.display = "grid";

            // Display Images
            document.getElementById("s2-img-t1").src = data.t1_image;
            document.getElementById("s2-img-t2").src = data.t2_image;
            document.getElementById("s2-img-mask").src = data.deforestation_mask;
            document.getElementById("s2-img-overlay").src = data.deforestation_overlay;

            document.getElementById("stage2-visuals").style.display = "grid";
        })
        .catch(err => {
            hideLoading();
            console.error(err);
            alert("Error running Stage 2 inference.");
        });
    });
});
