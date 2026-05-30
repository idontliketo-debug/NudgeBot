# ⚡ NudgeBot — Bot that helps in leading towards a better Understanding

NudgeBot is an AI tutor that nudges learners toward deeper understanding of NLP, ML, and AI concepts. Share what you think you know across RNNs, Transformers, LLMs, tokenization, embeddings, RAG, alignment, and more — and it will gently but firmly push your thinking further.

## 🏗️ Architecture

```
┌─────────────────────────────────┐
│         React Frontend          │ ← Port 3000
│  Chat UI · Topic Cards · Badges │
└──────────────┬──────────────────┘
               │ REST API
┌──────────────▼──────────────────┐
│         Flask Backend           │ ← Port 5000
│                                 │
│  ┌───────────────────────────┐  │
│  │     NLP Pipeline          │  │
│  │  • TF-IDF Intent Detect   │  │
│  │  • Sentence-BERT Similarity│ │
│  │  • VADER Sentiment         │ │
│  └───────────┬───────────────┘  │
│              │                  │
│  ┌───────────▼───────────────┐  │
│  │   Groq LLM (Llama 3.3)   │  │
│  │   Contrarian Responses     │ │
│  └───────────────────────────┘  │
└─────────────────────────────────┘
```

## 🚀 Quick Start (VS Code)

### Step 1: Get a Groq API Key (free)
1. Go to https://console.groq.com/keys
2. Create a free API key
3. Paste it in `backend/.env`

### Step 2: Backend Setup
Open a terminal in VS Code:
```bash
cd backend
pip install -r requirements.txt
```
Edit `.env` and add your Groq API key:
```
GROQ_API_KEY=gsk_your_actual_key_here
```
Start the server:
```bash
python app.py
```
You should see:
```
✅ Pipeline ready — 32 concepts loaded.
 * Running on http://127.0.0.1:5000
```

### Step 3: Frontend Setup
Open a **second** terminal in VS Code:
```bash
cd frontend
npm install
npm start
```
The app opens at http://localhost:3000

## 📁 Project Structure

```
nudgebot-project/
├── backend/
│   ├── app.py              # Flask API + NLP pipeline + Groq LLM
│   ├── requirements.txt    # Python dependencies
│   └── .env                # GROQ_API_KEY goes here
├── frontend/
│   ├── package.json        # React dependencies (proxies to :5000)
│   ├── public/
│   │   └── index.html
│   └── src/
│       ├── App.js          # Main React app with API integration
│       ├── App.css          # Dark editorial theme
│       └── index.js         # React entry point
└── README.md
```

## 🔬 NLP Features (from the notebook)

| Feature | Technique | Purpose |
|---|---|---|
| Intent Detection | TF-IDF + Sentence-BERT fallback | Maps questions → 32 RNN concepts |
| Text Similarity | Cosine similarity on SBERT embeddings | CORRECT/INCORRECT classification |
| Sentiment Analysis | VADER compound score | Detects confident/uncertain/neutral tone |
| LLM Response | Groq (Llama 3.3 70B) | Generates contrarian teaching responses |

## 🔌 API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/api/health` | Backend status check |
| POST | `/api/ask` | Send question + understanding, get AI response |
| POST | `/api/reset` | Clear conversation history for a session |
| GET | `/api/concepts` | List all 32 RNN concepts |

### Example Request
```bash
curl -X POST http://localhost:5000/api/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "What is LSTM?", "understanding": "LSTM uses gates to control memory", "session_id": "test1"}'
```

## 🎓 Covered Concepts (32 total)
RNN Basics · Hidden State · LSTM · GRU · Cell State · Gates (Forget/Input/Output/Reset/Update) · BPTT · Vanishing/Exploding Gradients · Gradient Clipping · Seq2Seq · Attention (Bahdanau/Luong) · Teacher Forcing · Bidirectional RNN · Deep RNN · Embeddings · Perplexity · Dropout · RNN vs Transformer · and more.

## 🛠️ Troubleshooting

**Backend won't start?** Make sure you have Python 3.9+ and run `pip install -r requirements.txt`.

**"GROQ_API_KEY not configured"?** Edit `backend/.env` with your real key from https://console.groq.com/keys.

**Frontend can't reach backend?** The `package.json` proxy forwards `/api/*` calls to port 5000. Make sure both servers are running.

**First request is slow?** The Sentence-BERT model loads on startup (~2-3 seconds). After that, responses are fast.
