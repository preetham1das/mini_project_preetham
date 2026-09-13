import { useState } from "react";
import "./App.css";

function App() {
  const [mode, setMode] = useState("forest"); // 'forest' or 'temporal'

  const [file, setFile] = useState(null);
  const [preview, setPreview] = useState(null);

  const [fileT1, setFileT1] = useState(null);
  const [previewT1, setPreviewT1] = useState(null);
  const [fileT2, setFileT2] = useState(null);
  const [previewT2, setPreviewT2] = useState(null);

  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);

  const handleModeChange = (newMode) => {
    setMode(newMode);
    setResult(null);
    setFile(null);
    setPreview(null);
    setFileT1(null);
    setPreviewT1(null);
    setFileT2(null);
    setPreviewT2(null);
  };

  const handleFileChange = (event) => {
    const selectedFile = event.target.files[0];
    if (!selectedFile) return;
    setFile(selectedFile);
    setPreview(URL.createObjectURL(selectedFile));
    setResult(null);
  };

  const handleFileT1Change = (event) => {
    const selectedFile = event.target.files[0];
    if (!selectedFile) return;
    setFileT1(selectedFile);
    setPreviewT1(URL.createObjectURL(selectedFile));
    setResult(null);
  };

  const handleFileT2Change = (event) => {
    const selectedFile = event.target.files[0];
    if (!selectedFile) return;
    setFileT2(selectedFile);
    setPreviewT2(URL.createObjectURL(selectedFile));
    setResult(null);
  };

  const analyzeImage = async () => {
    if (!file) return;
    setLoading(true);
    setResult(null);
    const formData = new FormData();
    formData.append("file", file);
    try {
      const response = await fetch("http://127.0.0.1:8000/predict", {
        method: "POST",
        body: formData,
      });
      const data = await response.json();
      if (!response.ok || !data.success) {
        throw new Error(data.error || "Prediction failed");
      }
      setResult(data);
    } catch (error) {
      alert(error.message);
    } finally {
      setLoading(false);
    }
  };

  const analyzeDeforestation = async () => {
    if (!fileT1 || !fileT2) return;
    setLoading(true);
    setResult(null);
    const formData = new FormData();
    formData.append("before_image", fileT1);
    formData.append("after_image", fileT2);
    try {
      const response = await fetch("http://127.0.0.1:8000/predict-temporal", {
        method: "POST",
        body: formData,
      });
      const data = await response.json();
      if (!response.ok || !data.success) {
        throw new Error(data.error || "Prediction failed");
      }
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
          <button
            className={mode === "forest" ? "active" : ""}
            onClick={() => handleModeChange("forest")}
          >
            Forest Cover
          </button>
          <button
            className={mode === "temporal" ? "active" : ""}
            onClick={() => handleModeChange("temporal")}
          >
            Deforestation Detection
          </button>
        </div>
      </header>

      <main>
        {mode === "forest" && (
          <>
            <section className="upload-section">
              <h2>Upload Satellite Image</h2>
              <label className="upload-box">
                <input
                  type="file"
                  accept="image/*,.tif,.tiff"
                  onChange={handleFileChange}
                />
                {file ? <span>{file.name}</span> : <span>Click to choose a satellite image</span>}
              </label>
              {file && (
                <button onClick={analyzeImage} disabled={loading}>
                  {loading ? "Analyzing..." : "Analyze Image"}
                </button>
              )}
            </section>

            {preview && (
              <section className="results">
                <div className="image-card">
                  <h2>Original Image</h2>
                  <img
                    src={result && result.original_image ? `data:image/png;base64,${result.original_image}` : preview}
                    alt="Uploaded satellite"
                  />
                </div>
                {result && (
                  <>
                    <div className="image-card">
                      <h2>Forest Segmentation</h2>
                      <img
                        src={`data:image/png;base64,${result.mask}`}
                        alt="Forest segmentation"
                      />
                    </div>
                    <div className="percentage-card">
                      <h2>Forest Coverage</h2>
                      <div className="percentage">{result.forest_percentage}%</div>
                      <p>Estimated forest area in the image</p>
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
              <h2>Upload Temporal Images</h2>
              <div className="temporal-uploads" style={{display: 'flex', gap: '20px', justifyContent: 'center'}}>
                <div className="upload-col">
                  <h3>Before Image (T1)</h3>
                  <label className="upload-box">
                    <input
                      type="file"
                      accept="image/*,.tif,.tiff"
                      onChange={handleFileT1Change}
                    />
                    {fileT1 ? <span>{fileT1.name}</span> : <span>Click to choose T1 image</span>}
                  </label>
                </div>
                <div className="upload-col">
                  <h3>After Image (T2)</h3>
                  <label className="upload-box">
                    <input
                      type="file"
                      accept="image/*,.tif,.tiff"
                      onChange={handleFileT2Change}
                    />
                    {fileT2 ? <span>{fileT2.name}</span> : <span>Click to choose T2 image</span>}
                  </label>
                </div>
              </div>
              {fileT1 && fileT2 && (
                <button onClick={analyzeDeforestation} disabled={loading} style={{marginTop: '20px'}}>
                  {loading ? "Analyzing..." : "Analyze Deforestation"}
                </button>
              )}
            </section>

            {(previewT1 || previewT2) && (
              <section className="results">
                {previewT1 && (
                  <div className="image-card">
                    <h2>Before Image (T1)</h2>
                    <img
                      src={result && result.before_image ? `data:image/png;base64,${result.before_image}` : previewT1}
                      alt="T1 satellite"
                    />
                  </div>
                )}
                {previewT2 && (
                  <div className="image-card">
                    <h2>After Image (T2)</h2>
                    <img
                      src={result && result.after_image ? `data:image/png;base64,${result.after_image}` : previewT2}
                      alt="T2 satellite"
                    />
                  </div>
                )}
                {result && (
                  <>
                    <div className="image-card">
                      <h2>Predicted Deforestation Mask</h2>
                      <img
                        src={`data:image/png;base64,${result.deforestation_mask}`}
                        alt="Deforestation Mask"
                      />
                    </div>
                    <div className="percentage-card">
                      <h2>Deforestation Extent</h2>
                      <div className="percentage">{result.deforestation_percentage}%</div>
                      <p>{result.changed_pixels} changed pixels detected.</p>
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
