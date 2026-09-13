import { useEffect, useRef, useState } from "react";
import ReactMarkdown from "react-markdown";

function CashewChatbot({ diagnosis, confidence, onClose }) {
  const [messages, setMessages] = useState([
    {
      sender: "bot",
      text: "Hi! I'm Cashew AI 🌿\n\nI can help you understand this diagnosis.",
    },
  ]);

  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);

  const messagesEndRef = useRef(null);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({
      behavior: "smooth",
    });
  }, [messages, loading]);

  // -----------------------------------------
  // SEND MESSAGE
  // -----------------------------------------
  const sendMessage = async (messageText = input) => {
    const trimmedMessage = messageText.trim();

    if (!trimmedMessage || loading) return;

    setMessages((prev) => [
      ...prev,
      {
        sender: "user",
        text: trimmedMessage,
      },
    ]);

    setInput("");
    setLoading(true);

    try {
      const response = await fetch("http://localhost:5000/api/chat", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          message: trimmedMessage,
          diagnosis: diagnosis,
          confidence: confidence,
        }),
      });

      const data = await response.json();

      if (!response.ok) {
        throw new Error(data.error || "Unable to get response");
      }

      setMessages((prev) => [
        ...prev,
        {
          sender: "bot",
          text:
            data.reply ||
            "Sorry, I couldn't generate a response right now.",
        },
      ]);
    } catch (error) {
      console.error("Chatbot error:", error);

      setMessages((prev) => [
        ...prev,
        {
          sender: "bot",
          text:
            "Sorry, I couldn't connect to Cashew AI right now. Please make sure the Flask backend is running.",
        },
      ]);
    } finally {
      setLoading(false);
    }
  };

  // -----------------------------------------
  // ENTER KEY
  // -----------------------------------------
  const handleKeyDown = (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  };

  // -----------------------------------------
  // QUICK QUESTIONS
  // -----------------------------------------
  const quickQuestions = [
    "What is this pest?",
    "What are the symptoms?",
    "How can I prevent it?",
  ];

  return (
    <div className="cashew-chatbot">

      {/* =====================================
          HEADER
      ===================================== */}
      <div className="chatbot-header">

        <div className="chatbot-header-icon">
          🌿
        </div>

        <div className="chatbot-header-text">
          <h3>Ask Cashew AI</h3>
          <span>AI-powered cashew assistant</span>
        </div>

        <div className="chatbot-online">
          <span></span>
          Online
        </div>

        {/* EXIT BUTTON */}
        <button
          className="chatbot-close-btn"
          onClick={onClose}
          aria-label="Close Cashew AI"
          title="Close"
        >
          ✕
        </button>

      </div>

      {/* =====================================
          CHAT AREA
      ===================================== */}
      <div className="chatbot-messages">

        {/* DIAGNOSIS CARD */}
        {diagnosis && (
          <div className="chatbot-diagnosis-card">

            <div className="chatbot-diagnosis-icon">
              🌱
            </div>

            <div className="chatbot-diagnosis-content">

              <div className="chatbot-diagnosis-label">
                Detected diagnosis
              </div>

              <div className="chatbot-diagnosis-name">
                {diagnosis}
              </div>

              {confidence !== undefined &&
                confidence !== null && (
                  <div className="chatbot-confidence">
                    {confidence}% confidence
                  </div>
                )}

            </div>

          </div>
        )}

        {/* =====================================
            CHAT MESSAGES
        ===================================== */}
        {messages.map((message, index) => (

          <div
            key={index}
            className={`chatbot-message ${
              message.sender === "user"
                ? "chatbot-user-message"
                : "chatbot-bot-message"
            }`}
          >

            {/* BOT ICON */}
            {message.sender === "bot" && (
              <div className="chatbot-avatar">
                🌿
              </div>
            )}

            {/* MESSAGE BUBBLE */}
            <div className="chatbot-bubble">

              {message.sender === "bot" ? (
                <ReactMarkdown
                  components={{
                    h1: ({ children }) => (
                      <h4 className="chatbot-md-heading">
                        {children}
                      </h4>
                    ),

                    h2: ({ children }) => (
                      <h4 className="chatbot-md-heading">
                        {children}
                      </h4>
                    ),

                    h3: ({ children }) => (
                      <h4 className="chatbot-md-heading">
                        {children}
                      </h4>
                    ),

                    h4: ({ children }) => (
                      <h4 className="chatbot-md-heading">
                        {children}
                      </h4>
                    ),

                    p: ({ children }) => (
                      <p className="chatbot-md-paragraph">
                        {children}
                      </p>
                    ),

                    ul: ({ children }) => (
                      <ul className="chatbot-md-list">
                        {children}
                      </ul>
                    ),

                    ol: ({ children }) => (
                      <ol className="chatbot-md-list">
                        {children}
                      </ol>
                    ),

                    li: ({ children }) => (
                      <li className="chatbot-md-list-item">
                        {children}
                      </li>
                    ),

                    strong: ({ children }) => (
                      <strong className="chatbot-md-bold">
                        {children}
                      </strong>
                    ),
                  }}
                >
                  {message.text}
                </ReactMarkdown>
              ) : (
                <div className="chatbot-user-text">
                  {message.text}
                </div>
              )}

            </div>

          </div>

        ))}

        {/* =====================================
            TYPING INDICATOR
        ===================================== */}
        {loading && (
          <div className="chatbot-message chatbot-bot-message">

            <div className="chatbot-avatar">
              🌿
            </div>

            <div className="chatbot-bubble chatbot-typing">

              <span></span>
              <span></span>
              <span></span>

            </div>

          </div>
        )}

        <div ref={messagesEndRef} />

      </div>

      {/* =====================================
          QUICK QUESTIONS
      ===================================== */}
      {!loading && (
        <div className="chatbot-quick-questions">

          {quickQuestions.map((question, index) => (
            <button
              key={index}
              className="chatbot-quick-btn"
              onClick={() => sendMessage(question)}
            >
              {question}
            </button>
          ))}

        </div>
      )}

      {/* =====================================
          INPUT
      ===================================== */}
      <div className="chatbot-input-area">

        <input
          type="text"
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={handleKeyDown}
          placeholder="Ask about the diagnosis..."
          disabled={loading}
        />

        <button
          className="chatbot-send-btn"
          onClick={() => sendMessage()}
          disabled={!input.trim() || loading}
        >
          ➤
        </button>

      </div>

    </div>
  );
}

export default CashewChatbot;