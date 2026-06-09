import React, { useState, useRef, useEffect, useCallback } from 'react';
import 'bootstrap/dist/css/bootstrap.min.css';
import 'bootstrap-icons/font/bootstrap-icons.css';
import './App.css';

// ── UTILITIES ─────────────────────────────────────────────────────────────────
const API_BASE = process.env.REACT_APP_API_URL || 'http://localhost:5000';

function generateSessionId() {
  return 'sess_' + Date.now() + '_' + Math.random().toString(36).slice(2, 9);
}

// ── AVATARS ───────────────────────────────────────────────────────────────────
// A friendly, non-gendered "garden" set. One avatar is chosen per session (see
// App) and stays consistent for every user message in that conversation; a new
// one is selected on Reset. `rgb` drives each avatar's themed chip gradient/glow.
const AVATARS = [
  { emoji: '🥑', rgb: '124,179,66' },
  { emoji: '🍅', rgb: '239,83,80' },
  { emoji: '🥕', rgb: '255,152,0' },
  { emoji: '🍆', rgb: '171,71,188' },
  { emoji: '🌶️', rgb: '244,81,30' },
  { emoji: '🥦', rgb: '102,187,106' },
  { emoji: '🍄', rgb: '236,64,122' },
  { emoji: '🌽', rgb: '253,216,53' },
  { emoji: '🫐', rgb: '92,107,192' },
  { emoji: '🥝', rgb: '156,204,101' },
];

// Pick a random avatar index, optionally avoiding an immediate repeat.
function pickAvatar(exclude) {
  if (AVATARS.length <= 1) return 0;
  let i = Math.floor(Math.random() * AVATARS.length);
  while (exclude != null && i === exclude) {
    i = Math.floor(Math.random() * AVATARS.length);
  }
  return i;
}

// Build the themed chip styling for a user avatar from its rgb triplet.
function userAvatarStyle(rgb) {
  return {
    background: `linear-gradient(135deg, rgba(${rgb}, 0.24), rgba(${rgb}, 0.06))`,
    borderColor: `rgba(${rgb}, 0.42)`,
    boxShadow: `0 4px 14px rgba(${rgb}, 0.22), inset 0 1px 0 rgba(255, 255, 255, 0.08)`,
  };
}

// ── HEADER ────────────────────────────────────────────────────────────────────
function Header({ onNewChat, messageCount }) {
  return (
    <header className="app-header">
      <div className="header-inner">
        <div className="brand">
          <span className="brand-icon" aria-hidden="true">⚡</span>
          <span className="brand-name">NudgeBot</span>
          <span className="brand-tag">YOUR NLP UNDERSTANDING GUIDE</span>
        </div>
        <div className="header-actions">
          {messageCount > 0 && (
            <span className="turn-counter">
              <i className="bi bi-chat-square-dots"></i>
              {Math.floor(messageCount / 2)} turns
            </span>
          )}
          <button className="btn-new-chat" onClick={onNewChat} title="New Chat" aria-label="Start a new chat">
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
function ChatMessage({ message, index, avatar }) {
  const isUser = message.role === 'user';
  const av = avatar || AVATARS[0];

  return (
    <div className={`msg-row ${isUser ? 'msg-user' : 'msg-bot'}`} style={{ animationDelay: `${index * 0.05}s` }}>
      {!isUser && (
        <div className="avatar avatar-bot" role="img" aria-label="NudgeBot">⚡</div>
      )}
      <div className={`msg-bubble ${isUser ? 'bubble-user' : 'bubble-bot'}`}>
        {isUser ? (
          <>
            <div className="msg-block">
              <div className="msg-label">Question</div>
              <div className="msg-text">{message.question}</div>
            </div>
            <div className="msg-block">
              <div className="msg-label">My Understanding</div>
              <div className="msg-text msg-text-soft">{message.understanding}</div>
            </div>
          </>
        ) : (
          <>
            <AnalysisBadges analysis={message.analysis} />
            <div className="msg-text">{message.content}</div>
          </>
        )}
      </div>
      {isUser && (
        <div className="avatar avatar-user" style={userAvatarStyle(av.rgb)} role="img" aria-label="Your avatar">
          <span className="avatar-emoji">{av.emoji}</span>
        </div>
      )}
    </div>
  );
}

// ── LOADING INDICATOR ─────────────────────────────────────────────────────────
function LoadingIndicator() {
  return (
    <div className="msg-row msg-bot">
      <div className="avatar avatar-bot" role="img" aria-label="NudgeBot">⚡</div>
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
          aria-label="Send"
          disabled={isLoading || !question.trim() || !understanding.trim()}
        >
          {isLoading ? (
            <span className="spinner-border spinner-border-sm"></span>
          ) : (
            <i className="bi bi-send-fill"></i>
          )}
        </button>
      </form>
      <p className="input-hint">
        NudgeBot pushes back to sharpen your thinking — double-check anything critical.
      </p>
    </div>
  );
}

// ── SUGGESTED TOPICS ──────────────────────────────────────────────────────────
function SuggestedTopics({ onSelect }) {
  // Per-card accent hues (rgb triplets) — drive each card's glow + arrow via the
  // --topic-rgb custom property, cycling so adjacent cards never share a color.
  const HUES = [
    '255, 206, 77', '167, 139, 250', '94, 234, 212', '129, 212, 250',
    '244, 114, 182', '163, 230, 53', '251, 146, 60', '96, 165, 250',
    '52, 211, 153', '232, 121, 249', '250, 204, 21', '125, 211, 252',
  ];
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
          style={{ '--topic-rgb': HUES[i % HUES.length], animationDelay: `${i * 0.08}s` }}
        >
          <span className="topic-q-row">
            <span className="topic-q">{topic.question}</span>
            <span className="topic-arrow" aria-hidden="true"><i className="bi bi-arrow-right"></i></span>
          </span>
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
  const [avatarIndex, setAvatarIndex] = useState(() => pickAvatar());
  const [backendStatus, setBackendStatus] = useState('checking');
  const chatEndRef = useRef(null);

  const avatar = AVATARS[avatarIndex];

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
    setAvatarIndex(prev => pickAvatar(prev)); // fresh avatar each new conversation
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
        <div className={`status-bar ${backendStatus}`} role="status">
          <span className="status-dot" aria-hidden="true"></span>
          {backendStatus === 'checking' && <span>Connecting to backend…</span>}
          {backendStatus === 'disconnected' && (
            <span>Backend not reachable — run <code>python app.py</code> in the backend folder</span>
          )}
        </div>
      )}

      <main className="chat-area">
        {/* Welcome screen */}
        {messages.length === 0 && (
          <div className="welcome">
            <span className="welcome-eyebrow">AI-Powered NLP &amp; ML Tutor</span>
            <div className="welcome-orb" aria-hidden="true">
              <span className="orb-ring"></span>
              <span className="orb-ring orb-ring-2"></span>
              <div className="welcome-icon">🧠</div>
            </div>
            <h1>Nudge your <em className="hero-em">understanding</em> forward.</h1>
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
          <ChatMessage key={i} message={msg} index={i} avatar={avatar} />
        ))}
        {isLoading && <LoadingIndicator />}
        <div ref={chatEndRef} />
      </main>

      <InputForm onSubmit={handleSubmit} isLoading={isLoading} />
    </div>
  );
}

export default App;
