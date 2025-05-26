import React, { useState } from "react";

const Main = ({ onSubmit }) => {
  const [url, setUrl] = useState("");

  const handleClick = () => {
    onSubmit(url);
  };

  return (
    <div className="main">
      <h2>🔍 Enter a YouTube Video URL Below:</h2>
      <div className="input-container">
        <input
          type="text"
          placeholder="Paste the YouTube video URL here..."
          value={url}
          onChange={(e) => setUrl(e.target.value)}
        />
        <button onClick={handleClick}>Get Sentiment! 🚀</button>
      </div>
    </div>
  );
};

export default Main;
