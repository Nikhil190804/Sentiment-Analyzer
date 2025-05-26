import React, { useState } from "react";
import "./App.css";
import Header from "./components/Header";
import Main from "./components/Main";
import Footer from "./components/Footer";
import Toast from "./components/Toast";

function App() {
  const [toastMessage, setToastMessage] = useState("");

  const showToast = (message) => {
    setToastMessage(message);
    setTimeout(() => setToastMessage(""), 5000);
  };

  const validateUrl = (url) => {
    url = url.trim();
    return url !== "" && url.startsWith("https://") && url.length >= 10;
  };

  const handleSubmit = (videoUrl) => {
    if (!validateUrl(videoUrl)) {
      showToast("Invalid URL!!!");
      return;
    }

    fetch("https://sentiment-analyzer-5w4g.onrender.com/query", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ yt_url: videoUrl }),
    })
      .then((res) => res.json())
      .then((data) => console.log("Success:", data))
      .catch((error) => console.error("Error:", error));
  };

  return (
    <div className="App">
      <Header />
      <Toast message={toastMessage} />
      <Main onSubmit={handleSubmit} />
      <Footer />
    </div>
  );
}

export default App;
