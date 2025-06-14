import React, { useState } from "react";
import "./App.css";
import Header from "./components/Header";
import Main from "./components/Main";
import Footer from "./components/Footer";
import Toast from "./components/Toast";
import ResultsPage from "./components/ResultsPage";
import {Routes, Route} from "react-router-dom";
import ClipLoader from "react-spinners/ClipLoader";

function App() {
  const [toastMessage, setToastMessage] = useState("");
  const [analysisData, setAnalysisData] = useState(null);
  const [isLoading, setIsLoading] = useState(false);

  const showToast = (message) => {
    setToastMessage(message);
    setTimeout(() => setToastMessage(""), 5000);
  };

  const validateUrl = (url) => {
    url = url.trim();
    return url !== "" && url.startsWith("https://") && url.length >= 10;
  };

  const handleSubmit = (videoUrl, navigate) => {
    if (!validateUrl(videoUrl)) {
      showToast("Invalid URL!!!");
      return;
    }
    setIsLoading(true);
    fetch("http://127.0.0.1:8080/query", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ yt_url: videoUrl }),
    })
      .then((res) => res.json())
      .then((data) => {
        setIsLoading(false);
        setAnalysisData(data);
        navigate("/results");
      })
      .catch((error) => {
        setIsLoading(false);
        console.error("Error:", error);
        showToast("Error occurred during analysis.");
      });
  };

  return (
    <div className="App">
      <Header />
      <Toast message={toastMessage} />

      {isLoading ? (
        <div className="loader-wrapper">
          <ClipLoader size={50} color="#007bff" />
          <p>Analyzing video comments...</p>
        </div>
      ) : (
        <Routes>
          <Route path="/" element={<Main onSubmit={handleSubmit} />} />
          <Route path="/results" element={<ResultsPage data={analysisData} />} />
        </Routes>
      )}
      <Footer />
    </div>
  );
}

export default App