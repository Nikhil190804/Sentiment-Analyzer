import React from "react";


function decodeBase64Image(base64Str) {
  return `data:image/png;base64,${base64Str}`;
}

function ResultsPage({ data }) {
  if (!data) return <div>Loading...</div>;

  return (
    <div className="results-container">
      <h2>🔍 Word Cloud</h2>
      <img src={decodeBase64Image(data.word_cloud)} alt="Word Cloud" />

      <h2>🧠 Sentiment Analysis</h2>
      <pre>{data.sentiment_output}</pre>
      <img src={decodeBase64Image(data.sentiment_graph)} alt="Sentiment Pie Chart" />

      <h2>⚠️ Offensive Language Analysis</h2>
      <pre>{data.offensive_language_output}</pre>
      <img src={decodeBase64Image(data.offensive_language_graph)} alt="Offensive Language Pie Chart" />

      <h2>😊 Emoji Analysis</h2>
      <pre>{data.emoji_output}</pre>
      <img src={decodeBase64Image(data.emoji_graph)} alt="Emoji Pie Chart" />

      <h2>🧠 LLM Output</h2>
      <pre>{data.llm_output}</pre>
    </div>
  );
}

export default ResultsPage;
