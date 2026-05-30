import React, { useState, useRef, useEffect, useCallback } from 'react';
import 'bootstrap/dist/css/bootstrap.min.css';
import 'bootstrap-icons/font/bootstrap-icons.css';
import './App.css';

// ── UTILITIES ─────────────────────────────────────────────────────────────────
const API_BASE = process.env.REACT_APP_API_URL || 'http://localhost:5000';

function generateSessionId() {
  return 'sess_' + Date.now() + '_' + Math.random().toString(36).slice(2, 9);
}

// ── HEADER ────────────────────────────────────────────────────────────────────
function Header({ onNewChat, messageCount }) {
  return (
    <header className="app-header">
      <div className="header-inner">
        <div className="brand">
          <span className="brand-icon">⚡</span>
          <span className="brand-name">NudgeBot</span>
          <span className="brand-tag">YOUR NLP UNDERSTANDING GUIDE</span>
        </div>
        <div className="header-actions">
          {messageCount > 0 && (
            <span className="turn-counter">{Math.floor(messageCount / 2)} turns</span>
          )}
          <button className="btn-new-chat" onClick={onNewChat} title="New Chat">
            <i className="bi bi-arrow-counterclockwise"></i>
            <span>Reset</span>
          </button>
        </div>
      </div>
    </header>
  );
}

// ── ANALYSIS PILL ─────────────────────────────────────────────────────────────
function AnalysisBadges({ analysis }) {
  if (!analysis) return null;

  const labelClass = analysis.label === 'CORRECT' ? 'badge-correct' : 'badge-incorrect';
  const toneEmoji = { confident: '💪', uncertain: '🤔', neutral: '😐' };

  return (
    <div className="analysis-row">
      <span className={`analysis-badge ${labelClass}`}>
        {analysis.label === 'CORRECT' ? '✓' : '✗'} {analysis.label}
      </span>
      <span className="analysis-badge badge-concept">
        {analysis.concept?.replace(/_/g, ' ')}
      </span>
      <span className="analysis-badge badge-sim">
        {(analysis.similarity * 100).toFixed(0)}% match
      </span>
      <span className="analysis-badge badge-tone">
        {toneEmoji[analysis.tone] || '😐'} {analysis.tone}
      </span>
    </div>
  );
}

// ── CHAT MESSAGE ──────────────────────────────────────────────────────────────
function ChatMessage({ message, index }) {
  const isUser = message.role === 'user';

  return (
    <div className={`msg-row ${isUser ? 'msg-user' : 'msg-bot'}`} style={{ animationDelay: `${index * 0.05}s` }}>
      {!isUser && (
        <div className="avatar avatar-bot">⚡</div>
      )}
      <div className={`msg-bubble ${isUser ? 'bubble-user' : 'bubble-bot'}`}>
        {isUser ? (
          <>
            <div className="msg-label">Question</div>
            <div className="msg-text">{message.question}</div>
            <div className="msg-label" style={{ marginTop: '0.5rem' }}>My Understanding</div>
            <div className="msg-text">{message.understanding}</div>
          </>
        ) : (
          <>
            <AnalysisBadges analysis={message.analysis} />
            <div className="msg-text">{message.content}</div>
          </>
        )}
      </div>
      {isUser && (
        <div className="avatar avatar-user">
          <i className="bi bi-person-fill"></i>
        </div>
      )}
    </div>
  );
}

// ── LOADING INDICATOR ─────────────────────────────────────────────────────────
function LoadingIndicator() {
  return (
    <div className="msg-row msg-bot">
      <div className="avatar avatar-bot">⚡</div>
      <div className="msg-bubble bubble-bot loading-bubble">
        <div className="typing-dots">
          <span></span><span></span><span></span>
        </div>
        <span className="loading-text">Nudging your understanding...</span>
      </div>
    </div>
  );
}

// ── INPUT FORM ────────────────────────────────────────────────────────────────
function InputForm({ onSubmit, isLoading }) {
  const [question, setQuestion] = useState('');
  const [understanding, setUnderstanding] = useState('');
  const questionRef = useRef(null);

  const handleSubmit = (e) => {
    e.preventDefault();
    if (question.trim() && understanding.trim() && !isLoading) {
      onSubmit(question, understanding);
      setQuestion('');
      setUnderstanding('');
      questionRef.current?.focus();
    }
  };

  return (
    <div className="input-panel">
      <form onSubmit={handleSubmit} className="input-form">
        <div className="input-group-custom">
          <div className="input-field">
            <label><i className="bi bi-patch-question"></i> Question</label>
            <input
              ref={questionRef}
              type="text"
              placeholder="e.g., What is an LSTM?"
              value={question}
              onChange={(e) => setQuestion(e.target.value)}
              disabled={isLoading}
            />
          </div>
          <div className="input-field">
            <label><i className="bi bi-lightbulb"></i> Your Understanding</label>
            <textarea
              rows="2"
              placeholder="e.g., LSTM uses gates and a cell state to remember long-term dependencies..."
              value={understanding}
              onChange={(e) => setUnderstanding(e.target.value)}
              disabled={isLoading}
            />
          </div>
        </div>
        <button
          type="submit"
          className="btn-submit"
          disabled={isLoading || !question.trim() || !understanding.trim()}
        >
          {isLoading ? (
            <span className="spinner-border spinner-border-sm"></span>
          ) : (
            <i className="bi bi-send-fill"></i>
          )}
        </button>
      </form>
    </div>
  );
}

// ── SUGGESTED TOPICS ──────────────────────────────────────────────────────────
function SuggestedTopics({ onSelect }) {
  const topics = [
    { question: "What is an LSTM?", understanding: "LSTM uses gates to control memory flow" },
    { question: "Explain vanishing gradient", understanding: "Gradients become very small during backpropagation" },
    { question: "What is attention mechanism?", understanding: "Attention lets the decoder focus on relevant encoder states" },
    { question: "What is BERT?", understanding: "BERT is a bidirectional transformer pretrained with masked language modeling" },
    { question: "What is BPE tokenization?", understanding: "BPE merges frequent character pairs into subword tokens" },
    { question: "What is RAG?", understanding: "RAG retrieves documents and uses them to ground LLM generation" },
    { question: "What is LoRA?", understanding: "LoRA adapts models with small low-rank matrices instead of full fine-tuning" },
    { question: "What is nucleus sampling?", understanding: "Top-p sampling picks from tokens whose cumulative probability exceeds p" },
    { question: "What is RLHF?", understanding: "RLHF aligns LLMs using a reward model trained on human preferences" },
    { question: "Explain Word2Vec", understanding: "Word2Vec trains word vectors to predict context from center words" },
    { question: "What is a GRU?", understanding: "GRU is a simpler version of LSTM with two gates" },
    { question: "Explain teacher forcing", understanding: "Using ground truth as decoder input during training" },
  ];

  return (
    <div className="topics-grid">
      {topics.map((topic, i) => (
        <button
          key={i}
          className="topic-card"
          onClick={() => onSelect(topic.question, topic.understanding)}
          style={{ animationDelay: `${i * 0.08}s` }}
        >
          <span className="topic-q">{topic.question}</span>
          <span className="topic-u">{topic.understanding}</span>
        </button>
      ))}
    </div>
  );
}

// ── MAIN APP ──────────────────────────────────────────────────────────────────
function App() {
  const [messages, setMessages] = useState([]);
  const [isLoading, setIsLoading] = useState(false);
  const [sessionId, setSessionId] = useState(generateSessionId);
  const [backendStatus, setBackendStatus] = useState('checking');
  const chatEndRef = useRef(null);

  // Check backend health on mount
  useEffect(() => {
    fetch(`${API_BASE}/api/health`)
      .then(res => res.json())
      .then(() => setBackendStatus('connected'))
      .catch(() => setBackendStatus('disconnected'));
  }, []);

  // Auto-scroll
  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages, isLoading]);

  const handleNewChat = useCallback(async () => {
    try {
      await fetch(`${API_BASE}/api/reset`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ session_id: sessionId })
      });
    } catch (e) { /* ignore */ }
    setMessages([]);
    setSessionId(generateSessionId());
  }, [sessionId]);

  const handleSubmit = async (question, understanding) => {
    const userMessage = { role: 'user', question, understanding };
    setMessages(prev => [...prev, userMessage]);
    setIsLoading(true);

    try {
      const res = await fetch(`${API_BASE}/api/ask`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ question, understanding, session_id: sessionId })
      });
      const data = await res.json();

      setMessages(prev => [...prev, {
        role: 'assistant',
        content: data.response,
        analysis: data.analysis
      }]);
    } catch (error) {
      console.error('API Error:', error);
      setMessages(prev => [...prev, {
        role: 'assistant',
        content: '⚠️ Could not reach the backend. Make sure the Flask server is running on port 5000.',
        analysis: null
      }]);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="app-shell">
      <Header onNewChat={handleNewChat} messageCount={messages.length} />

      {/* Status bar */}
      {backendStatus !== 'connected' && (
        <div className={`status-bar ${backendStatus}`}>
          {backendStatus === 'checking' && '🔄 Connecting to backend...'}
          {backendStatus === 'disconnected' && (
            <>⚠️ Backend not reachable — run <code>python app.py</code> in the backend folder</>
          )}
        </div>
      )}

      <main className="chat-area">
        {/* Welcome screen */}
        {messages.length === 0 && (
          <div className="welcome">
            <div className="welcome-icon">🧠</div>
            <h1>Nudge your understanding forward.</h1>
            <p>
              Pick any NLP or ML concept — RNNs, Transformers, LLMs, tokenization, embeddings,
              retrieval, alignment — share what you think you know, and I'll push back to sharpen it.
              That's how real learning happens.
            </p>
            <SuggestedTopics onSelect={handleSubmit} />
            <div className="pipeline-info">
              <span><i className="bi bi-cpu"></i> Intent Detection</span>
              <span><i className="bi bi-bar-chart"></i> Text Similarity</span>
              <span><i className="bi bi-emoji-smile"></i> Sentiment Analysis</span>
              <span><i className="bi bi-robot"></i> Groq LLM</span>
            </div>
          </div>
        )}

        {/* Messages */}
        {messages.map((msg, i) => (
          <ChatMessage key={i} message={msg} index={i} />
        ))}
        {isLoading && <LoadingIndicator />}
        <div ref={chatEndRef} />
      </main>

      <InputForm onSubmit={handleSubmit} isLoading={isLoading} />
    </div>
  );
}

export default App;
