import { useEffect, useRef, useState } from "react";
import "./Dashboard.css";
import CashewChatbot from "./components/CashewChatbot";

function Dashboard() {
  const user = JSON.parse(localStorage.getItem("user")) || {};

  const [activePage, setActivePage] = useState("overview");
  const [selectedImage, setSelectedImage] = useState(null);
  const [preview, setPreview] = useState(null);
  const [prediction, setPrediction] = useState(null);
  const [history, setHistory] = useState([]);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [showChatbot, setShowChatbot] = useState(false);

  const fileInputRef = useRef(null);

  useEffect(() => {
    loadHistory();
  }, []);

  async function loadHistory() {
    if (!user.id) {
      return;
    }

    try {
      const response = await fetch(
        "http://localhost:5000/history/" + user.id
      );

      if (!response.ok) {
        return;
      }

      const data = await response.json();

      setHistory(data.history || []);
    } catch (error) {
      console.error("History error:", error);
    }
  }

  function handleImageSelect(event) {
    const file = event.target.files[0];

    if (!file) {
      return;
    }

    if (!file.type.startsWith("image/")) {
      alert("Please select an image file.");
      return;
    }

    setSelectedImage(file);
    setPreview(URL.createObjectURL(file));
    setPrediction(null);
    setShowChatbot(false);
  }

  async function handleAnalyze() {
    if (!selectedImage) {
      alert("Please select an image first.");
      return;
    }

    if (!user.id) {
      alert("User session not found. Please login again.");
      return;
    }

    setIsAnalyzing(true);
    setPrediction(null);
    setShowChatbot(false);

    const formData = new FormData();

    formData.append("image", selectedImage);
    formData.append("user_id", user.id);

    try {
      const response = await fetch(
        "http://localhost:5000/predict",
        {
          method: "POST",
          body: formData,
        }
      );

      const data = await response.json();

      if (!response.ok) {
        alert(data.message || "Prediction failed.");
        return;
      }

      setPrediction(data);

      await loadHistory();
    } catch (error) {
      console.error("Prediction error:", error);
      alert("Unable to connect to the backend server.");
    } finally {
      setIsAnalyzing(false);
    }
  }

  function openFileSelector() {
    if (fileInputRef.current) {
      fileInputRef.current.click();
    }
  }

  function handleLogout() {
    localStorage.removeItem("user");
    window.location.href = "/";
  }

  function getPredictionName() {
    if (!prediction) {
      return "Unknown";
    }

    return (
      prediction.prediction ||
      prediction.diagnosis ||
      "Unknown"
    );
  }

  function getPredictionConfidence() {
    if (!prediction) {
      return 0;
    }

    return prediction.confidence || 0;
  }

  const totalPredictions = history.length;

  const detectedCount = history.filter(function (item) {
    return (
      item.status === "detected" ||
      item.status === "Detected"
    );
  }).length;

  const healthyCount = history.filter(function (item) {
    return (
      item.prediction === "Healthy" ||
      item.prediction === "healthy"
    );
  }).length;

  return (
    <div className="dashboard-container">

      {/* SIDEBAR */}

      <aside className="dashboard-sidebar">

        <div className="dashboard-logo">

          <div className="logo-icon">
            🌿
          </div>

          <div>
            <h2>Cashew AI</h2>
            <span>Smart Farming</span>
          </div>

        </div>

        <nav className="dashboard-navigation">

          <button
            type="button"
            className={
              activePage === "overview"
                ? "dashboard-nav active"
                : "dashboard-nav"
            }
            onClick={() => setActivePage("overview")}
          >
            <span>🏠</span>
            Overview
          </button>

          <button
            type="button"
            className={
              activePage === "diagnosis"
                ? "dashboard-nav active"
                : "dashboard-nav"
            }
            onClick={() => setActivePage("diagnosis")}
          >
            <span>🔬</span>
            Diagnosis
          </button>

          <button
            type="button"
            className={
              activePage === "history"
                ? "dashboard-nav active"
                : "dashboard-nav"
            }
            onClick={() => setActivePage("history")}
          >
            <span>📋</span>
            History
          </button>

          <button
            type="button"
            className={
              activePage === "about"
                ? "dashboard-nav active"
                : "dashboard-nav"
            }
            onClick={() => setActivePage("about")}
          >
            <span>ℹ️</span>
            About
          </button>

        </nav>

        <div className="dashboard-sidebar-bottom">

          <button
            type="button"
            className="dashboard-logout"
            onClick={handleLogout}
          >
            <span>🚪</span>
            Logout
          </button>

        </div>

      </aside>


      {/* MAIN CONTENT */}

      <main className="dashboard-main">

        <header className="dashboard-header">

          <div>

            <h1>
              {activePage === "overview" && "Dashboard"}
              {activePage === "diagnosis" && "Cashew Diagnosis"}
              {activePage === "history" && "Prediction History"}
              {activePage === "about" && "About Cashew AI"}
            </h1>

            <p>

              {activePage === "overview" &&
                "Monitor and analyze your cashew plants"}

              {activePage === "diagnosis" &&
                "Upload a cashew leaf image for AI-powered analysis"}

              {activePage === "history" &&
                "View your previous cashew leaf predictions"}

              {activePage === "about" &&
                "Learn more about the Cashew AI system"}

            </p>

          </div>


          <div className="dashboard-user">

            <div className="dashboard-user-info">

              <strong>
                {user.name || "User"}
              </strong>

              <span>
                {user.email || "Cashew Farmer"}
              </span>

            </div>

            <div className="dashboard-avatar">
              {(user.name || "U").charAt(0).toUpperCase()}
            </div>

          </div>

        </header>


        {/* OVERVIEW */}

        {activePage === "overview" && (

          <section className="dashboard-content">

            <div className="dashboard-welcome">

              <h2>
                Welcome back, {user.name || "User"}! 🌿
              </h2>

              <p>
                Use Cashew AI to identify pests and diseases
                affecting your cashew plants.
              </p>

            </div>


            <div className="dashboard-stats">

              <div className="stat-card">

                <div className="stat-icon">
                  🔬
                </div>

                <div>
                  <span>Total Predictions</span>
                  <strong>{totalPredictions}</strong>
                </div>

              </div>


              <div className="stat-card">

                <div className="stat-icon">
                  ⚠️
                </div>

                <div>
                  <span>Issues Detected</span>
                  <strong>{detectedCount}</strong>
                </div>

              </div>


              <div className="stat-card">

                <div className="stat-icon">
                  🌱
                </div>

                <div>
                  <span>Healthy Results</span>
                  <strong>{healthyCount}</strong>
                </div>

              </div>

            </div>


            <div className="dashboard-card">

              <div className="card-header">

                <div>

                  <h2>AI Diagnosis</h2>

                  <p>
                    Upload a clear image of a cashew leaf
                  </p>

                </div>

              </div>


              <div className="upload-area">

                <input
                  ref={fileInputRef}
                  type="file"
                  accept="image/*"
                  onChange={handleImageSelect}
                  style={{ display: "none" }}
                />

                {!preview ? (

                  <button
                    type="button"
                    className="upload-box"
                    onClick={openFileSelector}
                  >

                    <div className="upload-icon">
                      📷
                    </div>

                    <h3>
                      Upload Cashew Leaf Image
                    </h3>

                    <p>
                      Click here to select an image
                    </p>

                    <span>
                      JPG, JPEG or PNG
                    </span>

                  </button>

                ) : (

                  <div className="image-preview-container">

                    <img
                      src={preview}
                      alt="Selected cashew leaf"
                      className="image-preview"
                    />

                    <button
                      type="button"
                      className="change-image-button"
                      onClick={openFileSelector}
                    >
                      Choose Another Image
                    </button>

                  </div>

                )}

              </div>


              {selectedImage && (

                <div className="analyze-section">

                  <p>
                    Selected image:{" "}
                    <strong>
                      {selectedImage.name}
                    </strong>
                  </p>

                  <button
                    type="button"
                    className="analyze-button"
                    onClick={handleAnalyze}
                    disabled={isAnalyzing}
                  >
                    {isAnalyzing
                      ? "Analyzing..."
                      : "Analyze Image"}
                  </button>

                </div>

              )}

            </div>


            {prediction && (

              <div className="dashboard-card prediction-card">

                <div className="card-header">

                  <div>

                    <h2>
                      Diagnosis Result
                    </h2>

                    <p>
                      AI analysis of your cashew leaf
                    </p>

                  </div>

                </div>


                <div className="prediction-result">

                  <div className="prediction-main">

                    <span className="prediction-label">
                      Prediction
                    </span>

                    <h2>
                      {getPredictionName()}
                    </h2>

                    <span>
                      Category:{" "}
                      {prediction.category || "Unknown"}
                    </span>

                  </div>


                  <div className="prediction-confidence">

                    <span>
                      Confidence
                    </span>

                    <strong>
                      {getPredictionConfidence()}%
                    </strong>

                  </div>

                </div>


                {prediction.status && (

                  <div className="prediction-status">
                    Status: {prediction.status}
                  </div>

                )}


                <button
                  type="button"
                  className="ask-cashew-ai-button"
                  onClick={() => setShowChatbot(true)}
                >
                  🌿 Ask Cashew AI
                </button>

              </div>

            )}

          </section>

        )}


        {/* DIAGNOSIS */}

        {activePage === "diagnosis" && (

          <section className="dashboard-content">

            <div className="dashboard-card">

              <div className="card-header">

                <div>

                  <h2>
                    Cashew Leaf Diagnosis
                  </h2>

                  <p>
                    Upload an image to identify possible
                    pests or diseases.
                  </p>

                </div>

              </div>


              <div className="upload-area">

                <input
                  ref={fileInputRef}
                  type="file"
                  accept="image/*"
                  onChange={handleImageSelect}
                  style={{ display: "none" }}
                />

                {!preview ? (

                  <button
                    type="button"
                    className="upload-box"
                    onClick={openFileSelector}
                  >

                    <div className="upload-icon">
                      📷
                    </div>

                    <h3>
                      Select Cashew Leaf Image
                    </h3>

                    <p>
                      Click to upload an image
                    </p>

                    <span>
                      JPG, JPEG or PNG
                    </span>

                  </button>

                ) : (

                  <div className="image-preview-container">

                    <img
                      src={preview}
                      alt="Cashew leaf preview"
                      className="image-preview"
                    />

                    <button
                      type="button"
                      className="change-image-button"
                      onClick={openFileSelector}
                    >
                      Choose Another Image
                    </button>

                  </div>

                )}

              </div>


              {selectedImage && (

                <div className="analyze-section">

                  <p>
                    Selected:{" "}
                    <strong>
                      {selectedImage.name}
                    </strong>
                  </p>

                  <button
                    type="button"
                    className="analyze-button"
                    onClick={handleAnalyze}
                    disabled={isAnalyzing}
                  >
                    {isAnalyzing
                      ? "Analyzing..."
                      : "Analyze Image"}
                  </button>

                </div>

              )}

            </div>


            {prediction && (

              <div className="dashboard-card prediction-card">

                <h2>
                  Diagnosis Result
                </h2>

                <div className="prediction-result">

                  <div className="prediction-main">

                    <span className="prediction-label">
                      Prediction
                    </span>

                    <h2>
                      {getPredictionName()}
                    </h2>

                    <span>
                      Category:{" "}
                      {prediction.category || "Unknown"}
                    </span>

                  </div>


                  <div className="prediction-confidence">

                    <span>
                      Confidence
                    </span>

                    <strong>
                      {getPredictionConfidence()}%
                    </strong>

                  </div>

                </div>


                <button
                  type="button"
                  className="ask-cashew-ai-button"
                  onClick={() => setShowChatbot(true)}
                >
                  🌿 Ask Cashew AI
                </button>

              </div>

            )}

          </section>

        )}


        {/* HISTORY */}

        {activePage === "history" && (

          <section className="dashboard-content">

            <div className="dashboard-card">

              <div className="card-header">

                <div>

                  <h2>
                    Prediction History
                  </h2>

                  <p>
                    Your previous cashew leaf analyses
                  </p>

                </div>

              </div>


              {history.length === 0 ? (

                <div className="empty-state">

                  <div className="empty-icon">
                    📋
                  </div>

                  <h3>
                    No predictions yet
                  </h3>

                  <p>
                    Your diagnosis history will appear here
                    after you analyze an image.
                  </p>

                </div>

              ) : (

                <div className="history-list">

                  {history.map(function (item, index) {

                    return (

                      <div
                        className="history-item"
                        key={item.id || index}
                      >

                        <div className="history-icon">
                          🔬
                        </div>

                        <div className="history-details">

                          <strong>
                            {item.prediction ||
                              item.diagnosis ||
                              "Unknown"}
                          </strong>

                          <span>
                            Category:{" "}
                            {item.category || "Unknown"}
                          </span>

                          {item.created_at && (

                            <small>
                              {item.created_at}
                            </small>

                          )}

                        </div>


                        <div className="history-confidence">

                          {item.confidence !== undefined &&
                          item.confidence !== null
                            ? item.confidence + "%"
                            : "N/A"}

                        </div>

                      </div>

                    );

                  })}

                </div>

              )}

            </div>

          </section>

        )}


        {/* ABOUT */}

        {activePage === "about" && (

          <section className="dashboard-content">

            <div className="dashboard-card">

              <div className="card-header">

                <div>

                  <h2>
                    About Cashew AI
                  </h2>

                  <p>
                    AI-powered pest and disease diagnosis
                    for cashew farming
                  </p>

                </div>

              </div>


              <div className="about-content">

                <p>
                  Cashew AI is an AI-powered system designed
                  to assist in identifying pests and diseases
                  affecting cashew plants from leaf images.
                </p>

                <p>
                  The system uses deep learning and computer
                  vision techniques to analyze uploaded images
                  and provide a predicted diagnosis along with
                  a confidence score.
                </p>


                <div className="about-features">

                  <div>

                    <span>🌿</span>

                    <strong>
                      Cashew Focused
                    </strong>

                    <p>
                      Designed specifically for cashew
                      farming applications.
                    </p>

                  </div>


                  <div>

                    <span>🤖</span>

                    <strong>
                      AI Powered
                    </strong>

                    <p>
                      Uses deep learning for image-based
                      diagnosis.
                    </p>

                  </div>


                  <div>

                    <span>📊</span>

                    <strong>
                      Confidence Score
                    </strong>

                    <p>
                      Provides an indication of prediction
                      confidence.
                    </p>

                  </div>


                  <div>

                    <span>💬</span>

                    <strong>
                      AI Assistant
                    </strong>

                    <p>
                      Allows users to ask questions about
                      the current diagnosis.
                    </p>

                  </div>

                </div>

              </div>

            </div>

          </section>

        )}

      </main>


      {/* CHATBOT */}

      {showChatbot && prediction && (

        <CashewChatbot
          diagnosis={getPredictionName()}
          confidence={getPredictionConfidence()}
          onClose={() => setShowChatbot(false)}
        />

      )}

    </div>
  );
}

export default Dashboard;