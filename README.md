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
Open a terminal in VS Code and create an isolated **virtual environment** so NudgeBot's
dependencies stay separate from your other Python projects:

**Windows (PowerShell):**
```powershell
cd backend
python -m venv venv
.\venv\Scripts\python.exe -m pip install -r requirements.txt
```

**macOS / Linux:**
```bash
cd backend
python3 -m venv venv
./venv/bin/python -m pip install -r requirements.txt
```

> ⏳ The first install downloads PyTorch + Transformers (a few hundred MB) — give it a few minutes.

Add your Groq API key to `backend/.env`:
```
GROQ_API_KEY=gsk_your_actual_key_here
```

Start the server with the venv's Python (no activation needed):
```powershell
# Windows (PowerShell)
.\venv\Scripts\python.exe app.py
```
```bash
# macOS / Linux
./venv/bin/python app.py
```

You should see:
```
✅ Pipeline ready — 112 concepts loaded.
 * Running on http://127.0.0.1:5000
```

> 💡 **Tip:** Running the venv's Python directly (`.\venv\Scripts\python.exe`) avoids PowerShell activation and execution-policy prompts. To activate the venv instead, run `.\venv\Scripts\Activate.ps1` (Windows) or `source venv/bin/activate` (macOS/Linux), then just use `python app.py`. The `venv/` folder is git-ignored, so it's never committed.

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
│   ├── venv/               # Virtual environment (git-ignored, created in Step 2)
│   └── .env                # GROQ_API_KEY goes here
├── frontend/
│   ├── package.json        # React dependencies (proxies to :5000)
│   ├── public/
│   │   └── index.html
│   └── src/
│       ├── App.js          # Main React app with API integration
│       ├── App.css          # "Luminous Lab" theme (aurora + glassmorphism)
│       └── index.js         # React entry point
└── README.md
```

## 🔬 NLP Features (from the notebook)

| Feature | Technique | Purpose |
|---|---|---|
| Intent Detection | TF-IDF + Sentence-BERT fallback | Maps questions → 112 NLP/ML concepts |
| Text Similarity | Cosine similarity on SBERT embeddings | CORRECT/INCORRECT classification |
| Sentiment Analysis | VADER compound score | Detects confident/uncertain/neutral tone |
| LLM Response | Groq (Llama 3.3 70B) | Generates contrarian teaching responses |

## 🔌 API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/api/health` | Backend status check |
| POST | `/api/ask` | Send question + understanding, get AI response |
| POST | `/api/reset` | Clear conversation history for a session |
| GET | `/api/concepts` | List all 112 NLP/ML concepts |

### Example Request
```bash
curl -X POST http://localhost:5000/api/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "What is LSTM?", "understanding": "LSTM uses gates to control memory", "session_id": "test1"}'
```

## 🎓 Covered Concepts (112 total)
RNN Basics · Hidden State · LSTM · GRU · Cell State · Gates (Forget/Input/Output/Reset/Update) · BPTT · Vanishing/Exploding Gradients · Gradient Clipping · Seq2Seq · Attention (Bahdanau/Luong) · Teacher Forcing · Bidirectional RNN · Deep RNN · Embeddings · Perplexity · Dropout · RNN vs Transformer · and more.

## 🛠️ Troubleshooting

**Backend won't start?** Make sure you created and installed the venv (Step 2). A `ModuleNotFoundError` (e.g. `No module named 'flask_cors'`) means dependencies are missing — re-run the install with the venv's Python from the `backend` folder: `.\venv\Scripts\python.exe -m pip install -r requirements.txt` (Windows) or `./venv/bin/python -m pip install -r requirements.txt` (macOS/Linux).

**"GROQ_API_KEY not configured"?** Edit `backend/.env` with your real key from https://console.groq.com/keys.

**Frontend can't reach backend?** The `package.json` proxy forwards `/api/*` calls to port 5000. Make sure both servers are running.

**First request is slow?** The Sentence-BERT model loads on startup (~2-3 seconds). After that, responses are fast.
