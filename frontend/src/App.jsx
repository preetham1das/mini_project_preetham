import { useState, useEffect } from "react";
import "./App.css";

function App() {
  const [mode, setMode] = useState("forest");

  const [samples, setSamples] = useState({ stage1_samples: [], stage2_samples: [] });

  // Stage 1 State
  const [file, setFile] = useState(null);
  const [preview, setPreview] = useState(null);
  const [samplePath, setSamplePath] = useState("");
  const [thresholdS1, setThresholdS1] = useState(0.5);

  // Stage 2 State
  const [fileT1, setFileT1] = useState(null);
  const [previewT1, setPreviewT1] = useState(null);
  const [fileT2, setFileT2] = useState(null);
  const [previewT2, setPreviewT2] = useState(null);
  const [samplePair, setSamplePair] = useState("");
  const [thresholdS2, setThresholdS2] = useState(0.5);

  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    fetch("http://127.0.0.1:8000/api/samples")
      .then(res => res.json())
      .then(data => setSamples(data))
      .catch(err => console.error("Failed to load samples", err));
  }, []);

  const handleModeChange = (newMode) => {
    setMode(newMode);
    setResult(null);
    setFile(null); setPreview(null); setSamplePath("");
    setFileT1(null); setPreviewT1(null);
    setFileT2(null); setPreviewT2(null);
    setSamplePair("");
  };

  const handleFileChange = (event) => {
    const selectedFile = event.target.files[0];
    if (!selectedFile) return;
    setFile(selectedFile);
    setPreview(URL.createObjectURL(selectedFile));
    setSamplePath("");
    setResult(null);
  };

  const handleSampleS1Change = (e) => {
    setSamplePath(e.target.value);
    setFile(null);
    setPreview(null);
    setResult(null);
  };

  const handleFileT1Change = (event) => {
    const selectedFile = event.target.files[0];
    if (!selectedFile) return;
    setFileT1(selectedFile);
    setPreviewT1(URL.createObjectURL(selectedFile));
    setSamplePair("");
    setResult(null);
  };

  const handleFileT2Change = (event) => {
    const selectedFile = event.target.files[0];
    if (!selectedFile) return;
    setFileT2(selectedFile);
    setPreviewT2(URL.createObjectURL(selectedFile));
    setSamplePair("");
    setResult(null);
  };

  const handleSamplePairChange = (e) => {
    setSamplePair(e.target.value);
    setFileT1(null); setPreviewT1(null);
    setFileT2(null); setPreviewT2(null);
    setResult(null);
  };

  const analyzeImage = async () => {
    if (!file && !samplePath) return;
    setLoading(true);
    setResult(null);
    const formData = new FormData();
    if (file) formData.append("file", file);
    if (samplePath) formData.append("sample_path", samplePath);
    formData.append("threshold", thresholdS1);

    try {
      const response = await fetch("http://127.0.0.1:8000/predict", {
        method: "POST",
        body: formData,
      });
      const data = await response.json();
      if (!response.ok || !data.success) throw new Error(data.error || "Prediction failed");
      setResult(data);
    } catch (error) {
      alert(error.message);
    } finally {
      setLoading(false);
    }
  };

  const analyzeDeforestation = async () => {
    if ((!fileT1 || !fileT2) && !samplePair) return;
    setLoading(true);
    setResult(null);
    const formData = new FormData();
    if (fileT1 && fileT2) {
      formData.append("before_image", fileT1);
      formData.append("after_image", fileT2);
    }
    if (samplePair) {
      const obj = JSON.parse(samplePair);
      formData.append("sample_path_a", obj.path_a);
      formData.append("sample_path_b", obj.path_b);
    }
    formData.append("threshold", thresholdS2);

    try {
      const response = await fetch("http://127.0.0.1:8000/predict-temporal", {
        method: "POST",
        body: formData,
      });
      const data = await response.json();
      if (!response.ok || !data.success) throw new Error(data.error || "Prediction failed");
      setResult(data);
    } catch (error) {
      alert(error.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="app">
      <header>
        <h1>🌳 Forest Cover Analyzer</h1>
        <p>AI-based forest segmentation & deforestation detection</p>
        <div className="mode-selector">
          <button className={mode === "forest" ? "active" : ""} onClick={() => handleModeChange("forest")}>
            Forest Cover
          </button>
          <button className={mode === "temporal" ? "active" : ""} onClick={() => handleModeChange("temporal")}>
            Deforestation Detection
          </button>
        </div>
      </header>

      <main>
        {mode === "forest" && (
          <>
            <section className="upload-section">
              <h2>Select or Upload Satellite Image</h2>
              <div style={{ marginBottom: "20px" }}>
                <select value={samplePath} onChange={handleSampleS1Change} style={{ padding: "10px", width: "100%", maxWidth: "400px" }}>
                  <option value="">-- Or choose a sample image --</option>
                  {samples.stage1_samples.map(s => (
                    <option key={s.path} value={s.path}>[{s.split}] {s.name}</option>
                  ))}
                </select>
              </div>
              <label className="upload-box">
                <input type="file" accept="image/*,.tif,.tiff" onChange={handleFileChange} />
                {file ? <span>{file.name}</span> : <span>Click to choose a custom satellite image</span>}
              </label>
              <div style={{ margin: "20px 0" }}>
                <label>Threshold: {thresholdS1}</label><br/>
                <input type="range" min="0.1" max="0.9" step="0.05" value={thresholdS1} onChange={(e) => setThresholdS1(parseFloat(e.target.value))} />
              </div>
              {(file || samplePath) && (
                <button onClick={analyzeImage} disabled={loading}>
                  {loading ? "Analyzing..." : "Analyze Image"}
                </button>
              )}
            </section>

            {(preview || result) && (
              <section className="results">
                <div className="image-card">
                  <h2>Original Image</h2>
                  <img src={result?.original_image ? `data:image/png;base64,${result.original_image}` : preview} alt="Satellite" />
                </div>
                {result && (
                  <>
                    <div className="image-card">
                      <h2>Forest Segmentation</h2>
                      <img src={`data:image/png;base64,${result.mask}`} alt="Forest Mask" />
                    </div>
                    <div className="percentage-card">
                      <h2>Forest Coverage</h2>
                      <div className="percentage">{result.forest_percentage}%</div>
                      <p>{result.forest_pixels.toLocaleString()} pixels / {result.total_pixels.toLocaleString()} total pixels</p>
                    </div>
                  </>
                )}
              </section>
            )}
          </>
        )}

        {mode === "temporal" && (
          <>
            <section className="upload-section">
              <h2>Select or Upload Temporal Pair</h2>
              <div style={{ marginBottom: "20px" }}>
                <select value={samplePair} onChange={handleSamplePairChange} style={{ padding: "10px", width: "100%", maxWidth: "400px" }}>
                  <option value="">-- Or choose a sample pair --</option>
                  {samples.stage2_samples.map(s => (
                    <option key={s.id} value={JSON.stringify(s)}>[{s.split}] {s.name}</option>
                  ))}
                </select>
              </div>
              <div className="temporal-uploads" style={{ display: 'flex', gap: '20px', justifyContent: 'center' }}>
                <div className="upload-col">
                  <h3>Before Image (T1)</h3>
                  <label className="upload-box">
                    <input type="file" accept="image/*,.tif,.tiff" onChange={handleFileT1Change} />
                    {fileT1 ? <span>{fileT1.name}</span> : <span>Click to choose T1 image</span>}
                  </label>
                </div>
                <div className="upload-col">
                  <h3>After Image (T2)</h3>
                  <label className="upload-box">
                    <input type="file" accept="image/*,.tif,.tiff" onChange={handleFileT2Change} />
                    {fileT2 ? <span>{fileT2.name}</span> : <span>Click to choose T2 image</span>}
                  </label>
                </div>
              </div>
              <div style={{ margin: "20px 0" }}>
                <label>Threshold: {thresholdS2}</label><br/>
                <input type="range" min="0.1" max="0.9" step="0.05" value={thresholdS2} onChange={(e) => setThresholdS2(parseFloat(e.target.value))} />
              </div>
              {((fileT1 && fileT2) || samplePair) && (
                <button onClick={analyzeDeforestation} disabled={loading} style={{ marginTop: '20px' }}>
                  {loading ? "Analyzing..." : "Analyze Deforestation"}
                </button>
              )}
            </section>

            {(previewT1 || previewT2 || result) && (
              <section className="results">
                {(previewT1 || result) && (
                  <div className="image-card">
                    <h2>Before Image (T1)</h2>
                    <img src={result?.before_image ? `data:image/png;base64,${result.before_image}` : previewT1} alt="T1 Satellite" />
                  </div>
                )}
                {(previewT2 || result) && (
                  <div className="image-card">
                    <h2>After Image (T2)</h2>
                    <img src={result?.after_image ? `data:image/png;base64,${result.after_image}` : previewT2} alt="T2 Satellite" />
                  </div>
                )}
                {result && (
                  <>
                    <div className="image-card">
                      <h2>Predicted Deforestation Mask</h2>
                      <img src={`data:image/png;base64,${result.deforestation_mask}`} alt="Deforestation Mask" />
                    </div>
                    <div className="percentage-card">
                      <h2>Deforestation Extent</h2>
                      <div className="percentage">{result.deforestation_percentage}%</div>
                      <p>{result.changed_pixels.toLocaleString()} changed pixels / {result.total_pixels.toLocaleString()} total pixels</p>
                    </div>
                  </>
                )}
              </section>
            )}
          </>
        )}
      </main>
    </div>
  );
}

export default App;
