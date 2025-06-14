import React, { useState } from "react";
import "./ResultsPage.css";

function decodeBase64Image(base64Str) {
  return `data:image/png;base64,${base64Str}`;
}

function ResultsPage({ data }) {
  const [showChat, setShowChat] = useState(false);
  const [messages, setMessages] = useState([]);
  const [userInput, setUserInput] = useState("");

  const handleSend = async () => {
    if (!userInput.trim()) return;
    const userMsg = { from: "user", text: userInput };
    setMessages((prev) => [...prev, userMsg]);
    setUserInput("");
    try {
      const response = await fetch("http://127.0.0.1:8080/question", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ "question": userInput }),
      });
      const result = await response.json();
      const botMsg = { from: "bot", text: result.answer };
      setMessages((prev) => [...prev, botMsg]);
    } catch (error) {
      const botMsg = { from: "bot", text: "Something went wrong!" };
      setMessages((prev) => [...prev, botMsg]);
    }
  };

  const closeChat = () => {
    setShowChat(false);
    setMessages([]);
    setUserInput("");
  };

  if (!data) return <div className="loading">Loading...</div>;

  return (
    <>
      <div className="results-container">
        <h1 className="main-title">Insights From the Comments</h1>

        <section className="section">
          <h2 className="section-title">🧾 Commonly Used Words</h2>
          <img
            className="section-image centered"
            src={decodeBase64Image(data.word_cloud)}
            alt="Word Cloud"
          />
        </section>

        <section className="section split-layout">
          <div className="split-text">
            <h2 className="section-title">💬 How Do People Feel About This?</h2>
            <div className="text-box">{data.sentiment_output}</div>
          </div>
          <div className="split-image">
            <img
              src={decodeBase64Image(data.sentiment_graph)}
              alt="Sentiment Analysis Graph"
            />
          </div>
        </section>

        <section className="section split-layout">
          <div className="split-text">
            <h2 className="section-title">🚫 Any Inappropriate Language?</h2>
            <div className="text-box">{data.offensive_language_output}</div>
          </div>
          <div className="split-image">
            <img
              src={decodeBase64Image(data.offensive_language_graph)}
              alt="Offensive Language Graph"
            />
          </div>
        </section>

        <section className="section split-layout">
          <div className="split-text">
            <h2 className="section-title">😊 Emotions Through Emojis</h2>
            <div className="text-box">{data.emoji_output}</div>
          </div>
          <div className="split-image">
            <img
              src={decodeBase64Image(data.emoji_graph)}
              alt="Emoji Analysis Graph"
            />
          </div>
        </section>

        <section className="section">
          <h2 className="section-title">🗣️ What Users Are Saying?</h2>
          <div className="text-box">{data.llm_output}</div>
        </section>
      </div>

      {/* Floating Chatbot Button */}
      <div className="chatbot-icon" onClick={() => setShowChat(true)}>
        💬
      </div>

      {/* Chat Popup */}
      {showChat && (
        <div className="chat-popup">
          <div className="chat-header">
            <span>Talk to YouTube Comments Via LLM</span>
            <button className="chat-close" onClick={closeChat}>
              x
            </button>
          </div>
          <div className="chat-messages">
            {messages.map((msg, idx) => (
              <div key={idx} className={`chat-msg ${msg.from}`}>
                {msg.text}
              </div>
            ))}
          </div>
          <div className="chat-input-area">
            <input
              type="text"
              placeholder="Type a message..."
              value={userInput}
              onChange={(e) => setUserInput(e.target.value)}
              onKeyDown={(e) => e.key === "Enter" && handleSend()}
            />
            <button onClick={handleSend}>Send</button>
          </div>
        </div>
      )}
    </>
  );
}

export default ResultsPage;
