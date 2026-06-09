"""
NudgeBot — Backend API
Flask server exposing the NLP pipeline + Groq LLM as REST endpoints.
Bot that helps in leading towards a better Understanding.
"""

import os
import sys
import numpy as np

# ── Windows console safety ──────────────────────────────────────────────────
# The startup logs below use emoji (🔄 ✅ ⚠️). On Windows, stdout defaults to the
# cp1252 code page when piped or redirected, which raises UnicodeEncodeError on
# those characters and crashes the server before it starts. Reconfiguring to
# UTF-8 (with replacement) makes logging robust everywhere. This affects only
# how console output is encoded — no application behavior changes.
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except (AttributeError, ValueError):
    pass
from flask import Flask, request, jsonify
from flask_cors import CORS
from sentence_transformers import SentenceTransformer, util
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)
CORS(app)

# ── GROQ CLIENT ────────────────────────────────────────────────────────────────
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    print("⚠️  WARNING: GROQ_API_KEY not set. Add it to backend/.env")
groq_client = Groq(api_key=GROQ_API_KEY) if GROQ_API_KEY else None

SYSTEM_PROMPT = """You are NudgeBot — a teaching assistant that nudges learners toward better understanding of NLP, ML, and AI concepts.

NudgeBot's meaning: "Bot that helps in leading towards a better Understanding."

RULE 1 — ALWAYS NUDGE IN TONE.
Never say "You're correct" or validate without expanding. Always push the learner further.
Open with a gentle challenge: "That's on the right track, but there's more to it...",
"You're not wrong, but you're missing something crucial...", "Almost — let's sharpen that idea..."

RULE 2 — TWO RESPONSE PATHS based on NLP label provided:
- If label is CORRECT: Acknowledge it briefly, then immediately challenge
  with a deeper viewpoint, edge case, or implication they haven't considered.
- If label is INCORRECT: Firmly but kindly correct the misconception, using the
  same angle the student approached it from.

RULE 3 — USE THE TONE SIGNAL.
If tone is 'confident', be more probing in your pushback.
If 'uncertain', be warmer but still nudge toward clarity.
If 'neutral', keep a balanced, inquisitive tone.

RULE 4 — CONVERSATIONAL MEMORY.
You remember the full conversation. If the student is following up on a previous
exchange, refer back to what was said. Build on prior nudges.

RULE 5 — EDUCATION DOMAIN ONLY.
Topics include: RNNs, LSTMs, GRUs, Seq2Seq, Transformers, BERT, GPT, T5, tokenization,
word embeddings, attention, pretraining, fine-tuning, RLHF, LoRA, RAG, decoding strategies,
NLP tasks, evaluation metrics, and modern LLM infrastructure.
If the question isn't about these topics, say:
"I only engage with NLP, ML, and AI academic concepts."

RULE 6 — CONCISE. 2-4 sentences. No bullet points."""


# ── KNOWLEDGE BASE ─────────────────────────────────────────────────────────────
KNOWLEDGE_BASE = {}

KNOWLEDGE_BASE["rnn_basics"] = {
    "aliases": ["recurrent neural network", "rnn", "what is rnn", "rnn architecture", "sequential model", "sequence model", "recurrent network"],
    "explanation": "A Recurrent Neural Network (RNN) is a neural network designed for sequential data. Unlike feedforward networks, RNNs have connections that loop back, allowing information to persist across timesteps via a hidden state. At each step t, the hidden state h_t is computed as h_t = tanh(W_h * h_{t-1} + W_x * x_t + b), combining the current input and the previous hidden state."
}
KNOWLEDGE_BASE["hidden_state"] = {
    "aliases": ["hidden state", "rnn memory", "h_t", "recurrent state", "internal memory rnn", "short term memory rnn", "state vector"],
    "explanation": "The hidden state h_t in an RNN is a fixed-size vector that summarizes information from all previous timesteps up to t. It acts as the network's working memory, updated at each step by combining the previous hidden state and the current input. In LSTMs, it represents short-term memory while the cell state handles long-term memory."
}
KNOWLEDGE_BASE["unrolling_rnn"] = {
    "aliases": ["unrolling rnn", "unfolding rnn", "rnn unrolled", "rnn through time", "rnn computational graph", "rnn timesteps"],
    "explanation": "Unrolling an RNN means expanding the recurrent computation graph across all timesteps, making it resemble a deep feedforward network where each layer corresponds to one timestep. This enables backpropagation through time (BPTT). The depth of the unrolled graph equals the sequence length, which is why long sequences cause gradient problems."
}
KNOWLEDGE_BASE["parameter_sharing"] = {
    "aliases": ["parameter sharing rnn", "shared weights rnn", "same weights each step", "weight reuse rnn", "rnn weight sharing"],
    "explanation": "RNNs use the same weight matrices at every timestep — this is called parameter sharing. It allows the model to generalize across different positions in a sequence and reduces parameters drastically. However, errors compound multiplicatively during backpropagation, contributing to vanishing and exploding gradients."
}
KNOWLEDGE_BASE["rnn_types"] = {
    "aliases": ["types of rnn", "one to many rnn", "many to one rnn", "many to many rnn", "rnn architectures types", "rnn variants"],
    "explanation": "RNNs can be structured in several configurations: one-to-one (standard network), one-to-many (image captioning), many-to-one (sentiment analysis), many-to-many equal length (POS tagging), and many-to-many different lengths (machine translation via encoder-decoder). The choice depends on the task structure."
}
KNOWLEDGE_BASE["bptt"] = {
    "aliases": ["backpropagation through time", "bptt", "rnn training algorithm", "how rnn is trained", "truncated bptt", "truncated backpropagation"],
    "explanation": "Backpropagation Through Time (BPTT) trains RNNs by unrolling the network across all timesteps and computing gradients using the chain rule. The gradient involves products of Jacobians across all timesteps. Truncated BPTT limits the number of timesteps to reduce memory and computation at the cost of not capturing very long-range dependencies."
}
KNOWLEDGE_BASE["vanishing_gradient"] = {
    "aliases": ["vanishing gradient", "gradient vanishes", "gradient becomes zero", "long term dependency problem", "gradient shrinks", "rnn forgets long sequences"],
    "explanation": "The vanishing gradient problem occurs in RNNs during BPTT when gradients are multiplied by the recurrent weight matrix repeatedly across timesteps. If singular values of the weight matrix are less than 1, gradients shrink exponentially for early timesteps, preventing the network from learning long-range dependencies. LSTMs and GRUs were designed to address this."
}
KNOWLEDGE_BASE["exploding_gradient"] = {
    "aliases": ["exploding gradient", "gradient explodes", "nan loss rnn", "gradient clipping", "unstable rnn training", "gradient norm"],
    "explanation": "Exploding gradients occur when repeated multiplication of recurrent weights causes gradients to grow exponentially during BPTT, leading to large weight updates and NaN loss. The standard fix is gradient clipping: if the gradient norm exceeds a threshold (typically 1.0 or 5.0), the gradient is rescaled. Unlike vanishing gradients, exploding gradients are easier to detect and fix."
}
KNOWLEDGE_BASE["gradient_clipping"] = {
    "aliases": ["gradient clipping", "clip gradients", "gradient norm clipping", "prevent exploding gradient", "max norm", "clipping threshold"],
    "explanation": "Gradient clipping rescales the gradient vector when its L2 norm exceeds a predefined threshold: if norm(g) > threshold, then g = g * (threshold / norm(g)). This preserves the gradient direction while controlling magnitude. It is standard in all modern RNN and transformer training pipelines."
}
KNOWLEDGE_BASE["lstm"] = {
    "aliases": ["lstm", "long short term memory", "lstm network", "what is lstm", "lstm architecture", "lstm vs rnn"],
    "explanation": "LSTM (Long Short-Term Memory), introduced by Hochreiter and Schmidhuber (1997), is an RNN variant designed to learn long-range dependencies. It introduces a cell state as long-term memory and three gates (forget, input, output) using sigmoid activations. The additive cell state update allows gradients to flow without vanishing over many timesteps."
}
KNOWLEDGE_BASE["cell_state"] = {
    "aliases": ["cell state", "lstm cell state", "c_t", "long term memory lstm", "conveyor belt lstm", "lstm memory cell"],
    "explanation": "The cell state c_t in an LSTM is a separate memory vector that runs through the entire sequence with only minor linear interactions, acting as a conveyor belt for long-term information. It is updated additively: c_t = f_t * c_{t-1} + i_t * g_t. This additive update allows gradients to flow back without vanishing. The cell state is filtered through the output gate to produce h_t."
}
KNOWLEDGE_BASE["forget_gate"] = {
    "aliases": ["forget gate", "lstm forget", "what to forget lstm", "f_t gate", "sigmoid forget", "cell state reset"],
    "explanation": "The forget gate f_t = sigmoid(W_f * [h_{t-1}, x_t] + b_f) decides what fraction of the previous cell state to retain. Values near 1 mean keep everything, near 0 mean forget completely. For example, in language modeling, the forget gate might reset subject-verb agreement memory when a sentence ends. It was added by Gers et al. (1999) and significantly improved performance."
}
KNOWLEDGE_BASE["input_gate"] = {
    "aliases": ["input gate", "lstm input gate", "i_t gate", "what to write lstm", "update gate lstm", "candidate cell state"],
    "explanation": "The input gate has two parts: i_t = sigmoid(W_i * [h_{t-1}, x_t]) controls how much of the candidate to write, and g_t = tanh(W_g * [h_{t-1}, x_t]) is the candidate cell update. The actual cell update is i_t * g_t. This gate determines what new information from the current input should be stored in the cell state."
}
KNOWLEDGE_BASE["output_gate"] = {
    "aliases": ["output gate", "lstm output gate", "o_t gate", "what to output lstm", "hidden state from cell state", "lstm output"],
    "explanation": "The output gate o_t = sigmoid(W_o * [h_{t-1}, x_t] + b_o) controls what part of the cell state is exposed as the hidden state: h_t = o_t * tanh(c_t). The LSTM can maintain information in the cell state without necessarily exposing it, selectively reading from the cell state to produce output at each timestep."
}
KNOWLEDGE_BASE["lstm_equations"] = {
    "aliases": ["lstm equations", "lstm formulas", "lstm math", "lstm forward pass", "lstm gate equations", "lstm computation"],
    "explanation": "Full LSTM forward pass: f_t = sigmoid(W_f[h_{t-1},x_t]+b_f), i_t = sigmoid(W_i[h_{t-1},x_t]+b_i), g_t = tanh(W_g[h_{t-1},x_t]+b_g), o_t = sigmoid(W_o[h_{t-1},x_t]+b_o), c_t = f_t*c_{t-1} + i_t*g_t, h_t = o_t*tanh(c_t). The key insight is that c_t updates additively, enabling gradient flow across many timesteps."
}
KNOWLEDGE_BASE["gru"] = {
    "aliases": ["gru", "gated recurrent unit", "what is gru", "gru architecture", "gru vs lstm", "simplified lstm", "cho 2014"],
    "explanation": "GRU (Gated Recurrent Unit), proposed by Cho et al. (2014), simplifies LSTM by merging the cell state and hidden state into one, and using only two gates: reset (r_t) and update (z_t). The update gate combines LSTMs forget and input gates. GRUs have fewer parameters and train faster, often matching LSTM performance on smaller datasets."
}
KNOWLEDGE_BASE["reset_gate"] = {
    "aliases": ["reset gate", "gru reset", "r_t gate", "gru forget", "what reset gate does", "gru previous hidden state"],
    "explanation": "The reset gate r_t = sigmoid(W_r * [h_{t-1}, x_t]) in a GRU controls how much of the previous hidden state to use when computing the candidate hidden state. When r_t is near 0, the unit ignores the previous state and resets — useful for starting a new sub-sequence. When r_t is near 1, the previous state is fully used, similar to a standard RNN."
}
KNOWLEDGE_BASE["update_gate_gru"] = {
    "aliases": ["update gate", "gru update gate", "z_t gate", "gru interpolation", "update gate gru", "gru memory control"],
    "explanation": "The update gate z_t = sigmoid(W_z * [h_{t-1}, x_t]) controls how much of the previous hidden state to carry forward: h_t = (1-z_t)*h_{t-1} + z_t*h_tilde. When z_t is near 1, the new candidate dominates; when near 0, the previous hidden state is preserved unchanged — effectively skipping the current timestep for long-term dependency handling."
}
KNOWLEDGE_BASE["gru_vs_lstm"] = {
    "aliases": ["gru vs lstm", "gru or lstm", "difference gru lstm", "when to use gru", "when to use lstm", "gru lstm comparison"],
    "explanation": "GRU has fewer parameters (no separate cell state, two gates vs three) making it faster to train and better on small datasets. LSTM has more expressive power due to the separate cell state and finer-grained control via three gates, often outperforming GRU on tasks requiring very long-range memory. GRU is preferred when speed matters, LSTM when maximum sequence modeling capacity is needed."
}
KNOWLEDGE_BASE["sequence_to_sequence"] = {
    "aliases": ["seq2seq", "sequence to sequence", "encoder decoder", "machine translation rnn", "seq2seq model", "encoder decoder rnn"],
    "explanation": "Seq2Seq models use an encoder RNN that reads the input sequence and compresses it into a fixed-size context vector (the final hidden state), and a decoder RNN that generates output from this vector. This architecture allows variable-length input and output sequences and is the foundation for machine translation, summarization, and dialogue systems."
}
KNOWLEDGE_BASE["context_vector"] = {
    "aliases": ["context vector", "encoder output vector", "bottleneck seq2seq", "fixed size representation", "encoder final state", "seq2seq bottleneck"],
    "explanation": "The context vector is the encoder's final hidden state in a seq2seq model, compressing the entire input sequence into a fixed-size vector. It initializes the decoder's hidden state. The bottleneck problem arises because all input information must fit in this single vector, causing information loss for long sequences. The attention mechanism was introduced to overcome this."
}
KNOWLEDGE_BASE["teacher_forcing"] = {
    "aliases": ["teacher forcing", "ground truth input decoder", "training seq2seq", "exposure bias", "scheduled sampling", "teacher forcing training"],
    "explanation": "Teacher forcing feeds the ground truth token from the previous timestep as decoder input during training, rather than the model's own prediction. This speeds up convergence but causes exposure bias — a mismatch between training (ground truth inputs) and inference (model's own predictions). Scheduled sampling gradually replaces ground truth with model predictions to address this."
}
KNOWLEDGE_BASE["attention_mechanism"] = {
    "aliases": ["attention", "attention mechanism", "bahdanau attention", "additive attention", "alignment scores", "attention weights", "context vector attention", "luong attention"],
    "explanation": "Attention (Bahdanau et al., 2014) allows the decoder to look at all encoder hidden states at each decoding step. Alignment scores e_{t,s} = score(h_t, h_s) are computed for each encoder state, normalized via softmax to get attention weights alpha_{t,s}, and the context vector is their weighted sum c_t = sum(alpha * h_s). This removes the fixed-size bottleneck and improves performance on long sequences."
}
KNOWLEDGE_BASE["bahdanau_vs_luong"] = {
    "aliases": ["bahdanau attention", "luong attention", "additive vs multiplicative attention", "dot product attention", "attention types"],
    "explanation": "Bahdanau (additive) attention computes alignment scores using a feed-forward network: score(h_t, h_s) = v^T * tanh(W_1*h_t + W_2*h_s). Luong (multiplicative) attention uses dot products: score = h_t^T * W * h_s. Luong attention is computationally simpler and faster. Bahdanau computes alignment before the decoder output; Luong uses it after."
}
KNOWLEDGE_BASE["bidirectional_rnn"] = {
    "aliases": ["bidirectional rnn", "birnn", "bidir rnn", "forward backward rnn", "context both directions", "bidirectional lstm", "bilstm"],
    "explanation": "A Bidirectional RNN runs two separate RNNs over the input — one forward and one backward — and concatenates their hidden states at each timestep. This gives the model access to both past and future context at every position, crucial for tasks like NER and POS tagging. BiRNNs cannot be used for generation tasks since future tokens are unknown at inference."
}
KNOWLEDGE_BASE["deep_rnn"] = {
    "aliases": ["deep rnn", "stacked rnn", "multi layer rnn", "stacked lstm", "hierarchical rnn", "deep recurrent network"],
    "explanation": "Deep RNNs stack multiple recurrent layers where the hidden state of one layer becomes the input to the next, with each layer learning increasingly abstract temporal representations. Stacking 2-4 LSTM layers is standard in practice. Dropout between layers (not within the recurrent connection) is used for regularization. Beyond 4 layers, residual connections are needed."
}
KNOWLEDGE_BASE["rnn_language_model"] = {
    "aliases": ["rnn language model", "rnn lm", "language modeling rnn", "next word prediction rnn", "rnn text generation", "character rnn"],
    "explanation": "An RNN language model predicts the probability of the next token given all previous tokens: P(w_t | w_1,...,w_{t-1}). The input token is embedded, fed to the RNN, and the output hidden state is projected through a softmax over the vocabulary. Training minimizes cross-entropy loss. Perplexity = exp(average cross-entropy) is the standard evaluation metric."
}
KNOWLEDGE_BASE["perplexity"] = {
    "aliases": ["perplexity", "language model evaluation", "ppl", "rnn perplexity", "how to evaluate language model", "cross entropy language model"],
    "explanation": "Perplexity measures how well a language model predicts a test sequence: PPL = exp(-1/N * sum(log P(w_t|context))). Lower perplexity means the model assigns higher probability to the test data. A perplexity of k means the model is as confused as if it chose uniformly from k words. It is the standard metric for comparing language models."
}
KNOWLEDGE_BASE["rnn_dropout"] = {
    "aliases": ["rnn dropout", "dropout lstm", "variational dropout", "recurrent dropout", "dropout rnn regularization", "how to regularize rnn"],
    "explanation": "Standard dropout hurts RNN training by disrupting memory across timesteps. Variational dropout by Gal and Ghahramani (2016) fixes this by applying the same dropout mask at every timestep within a sequence. Dropout should be applied only to non-recurrent connections (between layers), not within the recurrent step, unless using the variational formulation."
}
KNOWLEDGE_BASE["embedding_layer"] = {
    "aliases": ["word embedding rnn", "embedding layer", "input embedding", "word vector rnn", "one hot vs embedding", "rnn input representation"],
    "explanation": "RNNs typically receive word embeddings (dense vectors) rather than raw one-hot vectors as input, reducing dimensionality drastically (e.g., 50k-dim one-hot to 300-dim embedding). Embeddings are either trained from scratch with the RNN or initialized with pretrained vectors (Word2Vec, GloVe, FastText). The embedding matrix is learned during training via backpropagation."
}
KNOWLEDGE_BASE["rnn_vs_transformer"] = {
    "aliases": ["rnn vs transformer", "why transformers replaced rnn", "rnn limitations", "transformer better than rnn", "attention vs recurrence"],
    "explanation": "RNNs process sequences step-by-step (sequential computation), making them slow to train and unable to parallelize across timesteps. Transformers use self-attention to process all positions simultaneously, enabling full parallelism and handling very long-range dependencies more directly. However, RNNs are more memory-efficient for very long sequences and still used in streaming inference settings."
}
KNOWLEDGE_BASE["rnn_limitations"] = {
    "aliases": ["rnn limitations", "problems with rnn", "rnn disadvantages", "rnn weaknesses", "why rnn is hard to train", "rnn challenges"],
    "explanation": "Key RNN limitations: (1) Vanishing/exploding gradients make learning long-range dependencies difficult. (2) Sequential computation prevents parallelization, making training slow. (3) Fixed-size hidden state bottlenecks information capacity. (4) Difficult to capture hierarchical structure. LSTMs/GRUs address (1), attention addresses (3), and transformers address (2) and (4)."
}

# =====================================================================
# PART 5 — TOKENIZATION
# =====================================================================
KNOWLEDGE_BASE["tokenization"] = {
    "aliases": ["tokenization", "tokenizer", "what is tokenization", "split text into tokens", "nlp tokenization", "text preprocessing tokens"],
    "explanation": "Tokenization splits raw text into smaller units (tokens) that a model can process. Granularities range from character-level (very long sequences, small vocab) to word-level (large vocab, OOV problem) to subword-level (BPE, WordPiece, SentencePiece — the modern standard). Modern LLMs use subword tokenization to balance vocabulary size with the ability to handle any input string."
}
KNOWLEDGE_BASE["bpe"] = {
    "aliases": ["bpe", "byte pair encoding", "byte-pair encoding", "bpe tokenizer", "gpt tokenizer", "subword merge"],
    "explanation": "Byte Pair Encoding (BPE) starts with a character-level vocabulary and iteratively merges the most frequent adjacent pair of tokens until reaching a target vocabulary size. Originally a compression algorithm, it was adapted for NLP by Sennrich et al. (2016). BPE handles rare and unknown words by decomposing them into known subword units. GPT-2/3/4 use byte-level BPE which operates on UTF-8 bytes, guaranteeing no out-of-vocabulary tokens."
}
KNOWLEDGE_BASE["wordpiece"] = {
    "aliases": ["wordpiece", "wordpiece tokenizer", "bert tokenizer", "wordpiece bpe", "## tokens", "subword tokenizer bert"],
    "explanation": "WordPiece is the subword tokenization algorithm used by BERT and its variants. Like BPE, it starts from characters and merges, but uses a likelihood-based criterion: pairs are merged to maximize the likelihood of the training corpus rather than purely frequency. Continuation pieces are marked with '##' (e.g., 'playing' -> 'play', '##ing'). This signals to the model that the piece is a non-initial part of a word."
}
KNOWLEDGE_BASE["sentencepiece"] = {
    "aliases": ["sentencepiece", "sentencepiece tokenizer", "language agnostic tokenizer", "spm", "unigram tokenizer", "t5 tokenizer"],
    "explanation": "SentencePiece (Kudo & Richardson, 2018) is a language-agnostic tokenizer that treats input as a raw byte stream including whitespace (encoded as ▁), so it does not depend on word-level pre-tokenization. It supports both BPE and Unigram language model algorithms. Used by T5, ALBERT, XLNet, and most multilingual models because it handles languages without space separators (Chinese, Japanese) uniformly."
}
KNOWLEDGE_BASE["subword_tokenization"] = {
    "aliases": ["subword tokenization", "subword units", "why subword", "subword vs word", "out of vocabulary", "oov solution"],
    "explanation": "Subword tokenization splits words into smaller meaningful pieces, solving the out-of-vocabulary (OOV) problem of word-level tokenization while keeping sequence lengths shorter than character-level. Common words remain a single token; rare or compound words decompose into multiple subwords. This enables models to generalize across morphological variants and handle previously unseen words by composing known pieces."
}
KNOWLEDGE_BASE["special_tokens"] = {
    "aliases": ["special tokens", "cls token", "sep token", "pad token", "unk token", "bos eos", "mask token", "tokenizer special"],
    "explanation": "Special tokens carry structural meaning rather than lexical content. Common ones: [CLS] (classification, BERT's sentence embedding), [SEP] (separator between segments), [PAD] (padding to fixed length), [UNK] (unknown word fallback), [MASK] (masked language modeling target), <bos>/<eos> (beginning/end of sequence). The model learns to interpret these tokens through training and they are typically excluded from loss computation when used as padding."
}

# =====================================================================
# PART 6 — WORD EMBEDDINGS
# =====================================================================
KNOWLEDGE_BASE["word2vec"] = {
    "aliases": ["word2vec", "word 2 vec", "mikolov", "word vectors", "static embeddings", "distributed word representations"],
    "explanation": "Word2Vec (Mikolov et al., 2013) learns dense word vectors from large unlabeled corpora using shallow neural networks. It comes in two variants: CBOW predicts a center word from surrounding context, while Skip-gram predicts surrounding context from a center word. The learned vectors capture semantic and syntactic relationships, famously enabling vector arithmetic like king - man + woman ≈ queen. Trained with negative sampling or hierarchical softmax for efficiency."
}
KNOWLEDGE_BASE["cbow_skipgram"] = {
    "aliases": ["cbow", "skip gram", "skip-gram", "continuous bag of words", "cbow vs skipgram", "word2vec variants"],
    "explanation": "CBOW (Continuous Bag of Words) averages context word vectors to predict the center word — fast to train and works well on frequent words. Skip-gram does the reverse: predicts each context word from the center word, treating each (center, context) pair as a training example. Skip-gram is slower but produces better representations for rare words. Both use a sliding window over the training corpus."
}
KNOWLEDGE_BASE["glove"] = {
    "aliases": ["glove", "global vectors", "pennington glove", "stanford glove", "co occurrence embeddings", "matrix factorization embeddings"],
    "explanation": "GloVe (Pennington et al., 2014) learns word embeddings by factorizing a global word-word co-occurrence matrix. Unlike Word2Vec which uses local context windows, GloVe directly optimizes the dot product of word vectors to match the log of co-occurrence counts. It captures global statistics in one shot and often performs comparably to Word2Vec on analogy and similarity benchmarks."
}
KNOWLEDGE_BASE["fasttext"] = {
    "aliases": ["fasttext", "fast text", "subword embeddings", "character n gram embeddings", "facebook fasttext", "fasttext word vectors"],
    "explanation": "FastText (Bojanowski et al., 2017) extends Word2Vec by representing each word as a bag of character n-grams. The word vector is the sum of its n-gram vectors. This handles morphologically rich languages well, generates reasonable embeddings for out-of-vocabulary words by composing n-grams, and improves performance on rare words compared to Word2Vec."
}
KNOWLEDGE_BASE["contextual_embeddings"] = {
    "aliases": ["contextual embeddings", "elmo", "contextualized word vectors", "context dependent embeddings", "dynamic embeddings", "static vs contextual"],
    "explanation": "Contextual embeddings produce different vectors for the same word depending on its surrounding context, capturing polysemy that static embeddings (Word2Vec, GloVe) cannot. ELMo (Peters et al., 2018) used bidirectional LSTM language models; BERT and GPT use transformers. The word 'bank' gets different embeddings in 'river bank' vs 'investment bank.' These have largely replaced static embeddings in modern NLP."
}
KNOWLEDGE_BASE["embedding_evaluation"] = {
    "aliases": ["embedding evaluation", "word analogy", "word similarity", "wordsim353", "intrinsic evaluation embeddings", "embedding quality"],
    "explanation": "Word embeddings are evaluated intrinsically (analogy tasks like the famous king:queen::man:woman, similarity benchmarks like WordSim-353 or SimLex-999) and extrinsically (improvement on downstream tasks like NER or sentiment analysis). Intrinsic metrics are fast but don't always correlate with downstream performance, so extrinsic evaluation on real tasks is the gold standard."
}

# =====================================================================
# PART 7 — TRANSFORMER ARCHITECTURE
# =====================================================================
KNOWLEDGE_BASE["transformer"] = {
    "aliases": ["transformer", "attention is all you need", "vaswani 2017", "transformer architecture", "what is transformer", "transformer model"],
    "explanation": "The Transformer (Vaswani et al., 2017, 'Attention Is All You Need') replaces recurrence entirely with self-attention. The original model has an encoder-decoder structure where each block contains multi-head self-attention, position-wise feedforward layers, residual connections, and layer normalization. It enables full parallel computation across positions and has become the dominant architecture for NLP, vision, and multimodal models."
}
KNOWLEDGE_BASE["self_attention"] = {
    "aliases": ["self attention", "self-attention", "intra attention", "scaled dot product self attention", "transformer attention", "attention within sequence"],
    "explanation": "Self-attention lets every token attend to every other token in the same sequence, computing a weighted combination of all tokens' representations. Each token produces query (Q), key (K), and value (V) vectors via learned linear projections. Attention scores are computed as softmax(QK^T/sqrt(d_k)) and applied to V. This allows direct modeling of dependencies between any two positions regardless of distance."
}
KNOWLEDGE_BASE["scaled_dot_product_attention"] = {
    "aliases": ["scaled dot product attention", "attention formula", "qkv attention", "attention equation", "softmax attention", "scaling factor attention"],
    "explanation": "Scaled dot-product attention computes Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) V. The scaling factor sqrt(d_k) prevents the dot products from growing too large in magnitude when d_k is large, which would push softmax into regions with extremely small gradients. Without scaling, training becomes unstable for high-dimensional keys."
}
KNOWLEDGE_BASE["multi_head_attention"] = {
    "aliases": ["multi head attention", "multihead attention", "attention heads", "parallel attention", "mha", "why multiple heads"],
    "explanation": "Multi-head attention runs h independent attention operations in parallel with different learned Q/K/V projections, then concatenates the results and applies a final linear projection. Each head can attend to different aspects (syntax, semantics, position) at different scales. With model dimension d_model and h heads, each head operates in d_model/h dimensions, keeping total computation similar to single-head attention."
}
KNOWLEDGE_BASE["positional_encoding"] = {
    "aliases": ["positional encoding", "position embedding", "sinusoidal encoding", "transformer position", "sequence order transformer", "absolute position"],
    "explanation": "Self-attention is permutation-invariant — it has no inherent notion of order — so transformers add positional encodings to input embeddings. The original transformer used fixed sinusoidal encodings: PE(pos, 2i) = sin(pos/10000^(2i/d)), PE(pos, 2i+1) = cos(pos/10000^(2i/d)). BERT uses learned positional embeddings. Modern alternatives like RoPE and ALiBi encode position in the attention mechanism itself."
}
KNOWLEDGE_BASE["rope"] = {
    "aliases": ["rope", "rotary position embedding", "rotary embeddings", "rotary positional encoding", "llama position encoding", "modern position encoding"],
    "explanation": "Rotary Position Embedding (RoPE), introduced by Su et al. (2021), encodes absolute position by rotating query and key vectors in 2D subspaces based on position, so their dot product naturally depends on relative position. RoPE generalizes well to longer sequences than seen in training and is used in LLaMA, GPT-NeoX, PaLM, and many recent LLMs, replacing learned absolute embeddings."
}
KNOWLEDGE_BASE["transformer_encoder"] = {
    "aliases": ["transformer encoder", "encoder block", "encoder only", "bert encoder", "encoder stack", "transformer encoder layer"],
    "explanation": "A transformer encoder block contains: multi-head self-attention (each token attends bidirectionally to all positions), residual connection plus layer norm, position-wise feedforward (two linear layers with GELU/ReLU), and another residual+norm. Encoder-only models like BERT stack many such blocks and are used for understanding tasks: classification, NER, embeddings."
}
KNOWLEDGE_BASE["transformer_decoder"] = {
    "aliases": ["transformer decoder", "decoder block", "decoder only", "gpt decoder", "decoder stack", "autoregressive decoder"],
    "explanation": "A transformer decoder block adds two things to the encoder block: masked self-attention (prevents attending to future positions for autoregressive generation) and, in encoder-decoder models, cross-attention over encoder outputs. Decoder-only models like GPT use only masked self-attention plus FFN. Encoder-decoder models like T5 and the original transformer use both."
}
KNOWLEDGE_BASE["masked_self_attention"] = {
    "aliases": ["masked self attention", "causal attention", "causal mask", "autoregressive mask", "lower triangular mask", "decoder mask"],
    "explanation": "Masked (causal) self-attention prevents each position from attending to future positions during training, ensuring the model can only condition on previously generated tokens. This is implemented by adding a triangular mask (-infinity above the diagonal) to attention scores before softmax. It is essential for autoregressive language models so that training matches the inference behavior of generating one token at a time."
}
KNOWLEDGE_BASE["cross_attention"] = {
    "aliases": ["cross attention", "encoder decoder attention", "cross-attention", "attention over encoder", "decoder cross attention"],
    "explanation": "Cross-attention is used in encoder-decoder transformers: queries come from the decoder, keys and values come from the encoder output. This lets the decoder attend over the input sequence at each generation step, similar to Bahdanau attention but in transformer form. T5, BART, and the original transformer use cross-attention; decoder-only models like GPT do not."
}
KNOWLEDGE_BASE["layer_norm"] = {
    "aliases": ["layer norm", "layer normalization", "layernorm", "transformer normalization", "pre norm vs post norm", "rmsnorm"],
    "explanation": "Layer normalization normalizes activations across the feature dimension for each token independently, then applies learned scale and shift. The original transformer used post-norm (norm after residual), but modern implementations prefer pre-norm (norm before sub-layer) for more stable training of deep models. RMSNorm is a simplified variant used in LLaMA that drops the mean centering."
}
KNOWLEDGE_BASE["feedforward_layer"] = {
    "aliases": ["feedforward transformer", "ffn", "mlp transformer", "position wise feedforward", "two layer mlp transformer", "transformer ffn"],
    "explanation": "Each transformer block contains a position-wise feedforward network (FFN): two linear layers with a non-linearity (ReLU originally, GELU in BERT/GPT, SwiGLU in LLaMA). The intermediate dimension is typically 4x the model dimension. The FFN is applied independently to each position and is where most of a transformer's parameters live — often outweighing attention parameters."
}

# =====================================================================
# PART 8 — PRETRAINED MODELS & TRAINING OBJECTIVES
# =====================================================================
KNOWLEDGE_BASE["bert"] = {
    "aliases": ["bert", "bidirectional encoder", "devlin 2018", "bert model", "bert architecture", "bert nlp"],
    "explanation": "BERT (Bidirectional Encoder Representations from Transformers, Devlin et al., 2018) is an encoder-only transformer pretrained with masked language modeling and next sentence prediction on large text corpora. Unlike previous unidirectional models, BERT conditions on both left and right context simultaneously. It is fine-tuned for downstream tasks (classification, NER, QA) by adding a small task-specific head on top of the pretrained encoder."
}
KNOWLEDGE_BASE["gpt"] = {
    "aliases": ["gpt", "generative pretrained transformer", "gpt model", "openai gpt", "gpt architecture", "decoder only model"],
    "explanation": "GPT (Generative Pretrained Transformer) is a decoder-only transformer trained on causal language modeling — predict the next token given all previous tokens. Successive versions (GPT, GPT-2, GPT-3, GPT-4) scaled up parameters and data, demonstrating that scale enables emergent in-context learning abilities. GPT models are autoregressive and well-suited for generation, dialogue, and few-shot learning via prompting."
}
KNOWLEDGE_BASE["t5"] = {
    "aliases": ["t5", "text to text transformer", "raffel 2019", "t5 model", "text-to-text", "encoder decoder pretrained"],
    "explanation": "T5 (Text-to-Text Transfer Transformer, Raffel et al., 2019) frames every NLP task as text-to-text: input is a string with a task prefix (e.g., 'translate English to German: ...') and output is a string. It uses an encoder-decoder transformer pretrained with a span corruption objective on the C4 dataset. The unified format makes multitask training natural and was influential for instruction-tuned models that followed."
}
KNOWLEDGE_BASE["roberta"] = {
    "aliases": ["roberta", "robustly optimized bert", "liu 2019", "roberta vs bert", "improved bert"],
    "explanation": "RoBERTa (Liu et al., 2019) is a re-trained BERT with better hyperparameters: trained longer on more data, removed next sentence prediction, used dynamic masking (different masks per epoch), and larger batch sizes with longer sequences. These changes alone substantially improved performance over BERT, demonstrating that BERT was significantly undertrained."
}
KNOWLEDGE_BASE["masked_language_modeling"] = {
    "aliases": ["masked language modeling", "mlm", "bert pretraining", "fill in the blank", "cloze task", "mask token prediction"],
    "explanation": "Masked Language Modeling (MLM) is BERT's main pretraining objective. Roughly 15% of input tokens are randomly replaced with [MASK] (or kept/randomly substituted), and the model predicts the original tokens from bidirectional context. This forces the encoder to build deep contextual representations. The 80/10/10 trick (80% mask, 10% random, 10% unchanged) reduces the train-test mismatch caused by [MASK] never appearing at fine-tuning time."
}
KNOWLEDGE_BASE["causal_language_modeling"] = {
    "aliases": ["causal language modeling", "clm", "next token prediction", "autoregressive language modeling", "gpt pretraining", "left to right lm"],
    "explanation": "Causal Language Modeling (CLM) trains the model to predict the next token given all previous tokens, maximizing P(w_t | w_1, ..., w_{t-1}). This is the pretraining objective for GPT-style decoder-only models. Because it produces a generative model directly, no special generation head is needed. It is the dominant paradigm for modern large language models."
}
KNOWLEDGE_BASE["next_sentence_prediction"] = {
    "aliases": ["next sentence prediction", "nsp", "bert nsp", "sentence pair task", "sentence relationship", "is next sentence"],
    "explanation": "Next Sentence Prediction (NSP) was a secondary BERT pretraining task where the model classifies whether sentence B actually follows sentence A in the corpus, or is a random sentence. It was intended to help downstream tasks involving sentence pairs. Subsequent work (RoBERTa, ALBERT) found NSP unhelpful or even harmful, and most modern models drop it in favor of longer-form objectives."
}
KNOWLEDGE_BASE["span_corruption"] = {
    "aliases": ["span corruption", "t5 pretraining", "denoising objective", "span masking", "infilling objective", "sentinel tokens"],
    "explanation": "Span corruption is T5's pretraining objective: random contiguous spans of tokens are replaced with unique sentinel tokens, and the decoder generates the missing spans concatenated with sentinels. This is more efficient than single-token masking because the decoder produces only the missing content, not the entire sequence. It generalizes BERT's MLM to sequence-to-sequence pretraining."
}
KNOWLEDGE_BASE["pretraining_finetuning"] = {
    "aliases": ["pretraining and finetuning", "transfer learning nlp", "pretrain then finetune", "fine tuning", "pretraining paradigm", "self supervised pretraining"],
    "explanation": "The pretrain-then-finetune paradigm dominated NLP from 2018 onwards: a model is pretrained on a large unlabeled corpus with a self-supervised objective (MLM, CLM, span corruption), then fine-tuned on a small labeled dataset for a downstream task. This transfers general linguistic knowledge to specialized tasks with much less labeled data. Modern variants include parameter-efficient fine-tuning, instruction tuning, and prompting."
}
KNOWLEDGE_BASE["bart"] = {
    "aliases": ["bart", "denoising autoencoder", "lewis 2019", "bart model", "bart pretraining", "noising bart"],
    "explanation": "BART (Lewis et al., 2019) is an encoder-decoder transformer pretrained as a denoising autoencoder: input text is corrupted with various noise functions (token masking, deletion, span masking, sentence permutation, document rotation) and the model reconstructs the original. It combines BERT's bidirectional encoder with GPT's autoregressive decoder, performing well on both understanding and generation tasks."
}
KNOWLEDGE_BASE["encoder_decoder_decoder_only"] = {
    "aliases": ["encoder only vs decoder only", "encoder decoder vs decoder only", "model architecture comparison", "bert vs gpt vs t5", "transformer variants"],
    "explanation": "Three main transformer families: (1) Encoder-only (BERT, RoBERTa) — bidirectional, best for understanding tasks like classification. (2) Decoder-only (GPT, LLaMA) — autoregressive, best for generation and increasingly dominant due to scaling laws. (3) Encoder-decoder (T5, BART) — separate processing of input and output, strong for translation and summarization. Decoder-only with sufficient scale has become the default for general-purpose LLMs."
}

# =====================================================================
# PART 9 — MODERN LLM TRAINING & ALIGNMENT
# =====================================================================
KNOWLEDGE_BASE["instruction_tuning"] = {
    "aliases": ["instruction tuning", "instruction finetuning", "sft", "supervised fine tuning", "flan", "instruction following"],
    "explanation": "Instruction tuning fine-tunes a pretrained LLM on a dataset of (instruction, response) pairs covering many tasks. After this, the model can follow natural language instructions on tasks it was not specifically trained for. FLAN, T0, and InstructGPT pioneered this approach. Instruction tuning is typically the first stage of producing a usable assistant model from a raw pretrained LLM, before RLHF."
}
KNOWLEDGE_BASE["rlhf"] = {
    "aliases": ["rlhf", "reinforcement learning from human feedback", "ppo language model", "instructgpt", "human preferences training", "reward model"],
    "explanation": "RLHF (Reinforcement Learning from Human Feedback) aligns LLMs with human preferences in three stages: (1) supervised fine-tuning on demonstrations, (2) train a reward model to predict human preference rankings between model outputs, (3) optimize the LLM against the reward model using PPO. Used in InstructGPT, ChatGPT, and Claude. Addresses the gap between next-token prediction (pretraining objective) and being helpful, harmless, and honest."
}
KNOWLEDGE_BASE["dpo"] = {
    "aliases": ["dpo", "direct preference optimization", "rafailov dpo", "alternative to rlhf", "preference optimization", "rlhf alternative"],
    "explanation": "Direct Preference Optimization (DPO, Rafailov et al., 2023) is a simpler alternative to RLHF that achieves the same alignment goal without an explicit reward model or RL training loop. DPO derives a closed-form loss directly on preference pairs (chosen, rejected), training the model with standard supervised learning. It is more stable and easier to implement than PPO-based RLHF and has become widely adopted."
}
KNOWLEDGE_BASE["in_context_learning"] = {
    "aliases": ["in context learning", "icl", "few shot prompting", "gpt3 in context", "learning from prompt", "demonstrations in prompt"],
    "explanation": "In-context learning (ICL) is the ability of large LLMs to perform new tasks by conditioning on a few demonstrations in the prompt, without any gradient updates. The model treats the demonstrations as part of its input context and pattern-matches to produce appropriate outputs. Discovered prominently in GPT-3 (Brown et al., 2020), it scales with model size and is the foundation of modern prompt-based usage."
}
KNOWLEDGE_BASE["few_shot_learning"] = {
    "aliases": ["few shot learning", "few-shot", "k shot prompting", "zero shot", "one shot", "few shot vs zero shot"],
    "explanation": "Zero-shot prompts ask the model to perform a task with only the instruction and no examples. One-shot includes a single example; few-shot includes several. Performance generally improves with more examples up to a point, but is bounded by the context window. Few-shot prompting works because large LLMs implicitly recognize the task pattern from examples and apply it to the new query, exploiting in-context learning."
}
KNOWLEDGE_BASE["chain_of_thought"] = {
    "aliases": ["chain of thought", "cot prompting", "reasoning prompting", "wei 2022", "step by step prompting", "lets think step by step"],
    "explanation": "Chain-of-thought (CoT, Wei et al., 2022) prompting elicits intermediate reasoning steps before the final answer. Including 'Let's think step by step' or providing few-shot examples with worked-out reasoning substantially improves performance on math, logic, and multi-step problems. CoT is an emergent capability that appears only in sufficiently large models. Variants include self-consistency (sample multiple chains and vote) and Tree of Thoughts."
}
KNOWLEDGE_BASE["prompt_engineering"] = {
    "aliases": ["prompt engineering", "prompting", "prompt design", "how to prompt llm", "prompt template", "system prompt"],
    "explanation": "Prompt engineering is the practice of crafting inputs to LLMs to elicit desired behavior. Effective techniques include: clear role/task descriptions, few-shot examples, structured output formats, chain-of-thought triggers, and explicit constraints. System prompts set persistent instructions for chat models. Prompt engineering is now a partial substitute for fine-tuning when adapting LLMs to specific tasks without retraining."
}
KNOWLEDGE_BASE["lora"] = {
    "aliases": ["lora", "low rank adaptation", "peft", "parameter efficient fine tuning", "hu 2021 lora", "adapter llm"],
    "explanation": "LoRA (Low-Rank Adaptation, Hu et al., 2021) fine-tunes large models by inserting small trainable low-rank matrices into linear layers while keeping original weights frozen. Instead of updating a d×d weight matrix W, LoRA learns W + BA where B is d×r and A is r×d with r << d. This reduces trainable parameters by orders of magnitude (often >99%), saves GPU memory, and allows multiple task-specific adapters to be swapped at inference."
}
KNOWLEDGE_BASE["peft"] = {
    "aliases": ["peft", "parameter efficient fine tuning", "adapters", "prefix tuning", "prompt tuning", "efficient finetuning"],
    "explanation": "Parameter-Efficient Fine-Tuning (PEFT) is a family of methods that adapt large pretrained models by updating only a small subset of parameters or adding small trainable modules. Methods include adapters (small bottleneck layers), prefix tuning (learnable prefixes for activations), prompt tuning (learnable soft prompts), and LoRA. PEFT is essential for fine-tuning huge models on commodity hardware."
}
KNOWLEDGE_BASE["quantization"] = {
    "aliases": ["quantization", "model quantization", "int8 quantization", "4 bit quantization", "gptq", "awq", "qlora"],
    "explanation": "Quantization reduces the precision of model weights and/or activations from 16/32-bit floats to 8-bit, 4-bit, or even lower integers, dramatically shrinking memory and compute. Post-training quantization (GPTQ, AWQ) compresses an existing model; quantization-aware training and QLoRA train with quantized weights. Modern 4-bit quantization preserves most of an LLM's quality at a quarter of the memory."
}
KNOWLEDGE_BASE["knowledge_distillation"] = {
    "aliases": ["knowledge distillation", "model distillation", "teacher student", "distilbert", "compress llm", "soft labels distillation"],
    "explanation": "Knowledge distillation trains a small 'student' model to mimic a larger 'teacher' by matching its output distribution (soft labels) rather than just hard ground-truth labels. The teacher's softened probabilities convey richer information about class relationships. DistilBERT distilled BERT down to 60% size with 97% of performance. Distillation is widely used to deploy compact specialized models from massive teachers."
}
KNOWLEDGE_BASE["mixture_of_experts"] = {
    "aliases": ["mixture of experts", "moe", "sparse models", "switch transformer", "mixtral", "expert routing"],
    "explanation": "Mixture-of-Experts (MoE) replaces dense feedforward layers with many parallel 'expert' networks plus a router that activates only a few experts per token. This decouples total parameters (huge) from compute per token (small), enabling much larger models at the same training cost. Switch Transformer, GLaM, and Mixtral popularized this approach in LLMs. Challenges include load balancing across experts and training stability."
}

# =====================================================================
# PART 10 — DECODING & GENERATION
# =====================================================================
KNOWLEDGE_BASE["greedy_decoding"] = {
    "aliases": ["greedy decoding", "argmax decoding", "greedy search", "deterministic generation", "greedy text generation"],
    "explanation": "Greedy decoding selects the highest-probability token at each step. It is fast and deterministic but myopic: locally optimal choices often lead to globally suboptimal sequences and produce repetitive, dull output. Greedy decoding is rarely used directly for open-ended generation but is appropriate when there is a single correct answer (e.g., classification cast as generation)."
}
KNOWLEDGE_BASE["beam_search"] = {
    "aliases": ["beam search", "beam decoding", "beam width", "k best decoding", "search for translation", "neural machine translation decoding"],
    "explanation": "Beam search keeps the top-k most probable partial sequences (beams) at each step, expanding all and re-pruning to k. With k=1 it reduces to greedy decoding; larger k explores more but costs more compute. Beam search is standard for tasks with a clear correct output (machine translation, summarization) but produces bland, repetitive text for open-ended generation, where sampling methods are preferred."
}
KNOWLEDGE_BASE["top_k_sampling"] = {
    "aliases": ["top k sampling", "top-k", "sampling top k", "k sampling generation", "fan top k"],
    "explanation": "Top-k sampling restricts sampling to the k tokens with highest probability at each step, renormalizing their probabilities and sampling from this truncated distribution. It cuts off the long tail of low-probability tokens that often produce incoherent text. Choosing k is task-dependent (typical values 40-100). It can over-truncate when the model is uncertain (many plausible options) and under-truncate when confident."
}
KNOWLEDGE_BASE["top_p_sampling"] = {
    "aliases": ["top p sampling", "nucleus sampling", "top-p", "holtzman 2019", "dynamic truncation", "p sampling"],
    "explanation": "Top-p (nucleus) sampling, introduced by Holtzman et al. (2019), samples from the smallest set of tokens whose cumulative probability exceeds p (e.g., 0.9). Unlike top-k, the sampling pool size adapts to the model's confidence: peaked distributions sample from few tokens, flat distributions sample from many. Nucleus sampling produces more diverse and coherent generations than top-k for most open-ended tasks."
}
KNOWLEDGE_BASE["temperature_sampling"] = {
    "aliases": ["temperature sampling", "temperature parameter", "softmax temperature", "temperature llm", "creative vs deterministic"],
    "explanation": "Temperature T rescales logits before softmax: p_i ∝ exp(z_i / T). T=1 is the original distribution; T<1 sharpens it (more deterministic, T→0 approaches greedy); T>1 flattens it (more random, more diverse but lower quality). Temperature is typically combined with top-k or top-p. Common settings: T=0 for factual tasks, T=0.7-1.0 for creative writing."
}
KNOWLEDGE_BASE["repetition_penalty"] = {
    "aliases": ["repetition penalty", "no repeat ngram", "frequency penalty", "presence penalty", "avoid repetition generation"],
    "explanation": "Repetition penalty discourages already-generated tokens by dividing their logits by a factor > 1 before sampling, reducing degenerate loops common in LLMs. Variants include frequency penalty (scales with token count) and presence penalty (binary: appeared or not). 'No-repeat n-gram' blocks any n-gram from appearing twice. These mitigations partially address the loops produced by greedy and beam search."
}
KNOWLEDGE_BASE["beam_search_problems"] = {
    "aliases": ["beam search problems", "beam search degeneration", "neural text degeneration", "bland output beam search", "why sample instead of beam"],
    "explanation": "Beam search optimizes for likelihood, but high-likelihood text in open-ended settings tends to be bland, repetitive, and sometimes incoherent — the 'neural text degeneration' problem (Holtzman et al., 2019). Human text is not maximally probable; it has natural variation. This is why sampling methods like nucleus sampling outperform beam search for creative generation, while beam search remains useful for translation where the target distribution is more constrained."
}

# =====================================================================
# PART 11 — NLP TASKS & EVALUATION
# =====================================================================
KNOWLEDGE_BASE["named_entity_recognition"] = {
    "aliases": ["named entity recognition", "ner", "entity extraction", "extract entities", "person organization location", "ner tagging"],
    "explanation": "Named Entity Recognition (NER) identifies and classifies named entities in text into categories like Person, Organization, Location, Date, Money. Modern NER systems use transformer encoders (BERT, RoBERTa) with a token classification head, predicting a tag for each subword. The BIO tagging scheme marks Beginning, Inside, and Outside of entities. CoNLL-2003 is the canonical English NER benchmark."
}
KNOWLEDGE_BASE["pos_tagging"] = {
    "aliases": ["pos tagging", "part of speech", "pos", "grammatical tagging", "noun verb adjective", "morphosyntactic tagging"],
    "explanation": "Part-of-speech (POS) tagging assigns each word a grammatical category (noun, verb, adjective, etc.) based on its definition and context. The Penn Treebank tag set (45 tags) and Universal Dependencies tags are standard. Modern POS taggers use BiLSTMs or transformer encoders with token classification heads, achieving >97% accuracy on English. POS tags are often used as features for downstream tasks like parsing."
}
KNOWLEDGE_BASE["sentiment_analysis"] = {
    "aliases": ["sentiment analysis", "opinion mining", "sentiment classification", "polarity detection", "positive negative review"],
    "explanation": "Sentiment analysis classifies text by its expressed sentiment — typically positive/negative/neutral, sometimes fine-grained (1-5 stars) or aspect-based (sentiment toward specific entities). Modern systems fine-tune transformer encoders on labeled corpora like SST-2 (Stanford Sentiment Treebank), IMDB reviews, or Amazon reviews. It is among the most widely deployed NLP tasks for product feedback analysis."
}
KNOWLEDGE_BASE["machine_translation"] = {
    "aliases": ["machine translation", "mt", "neural machine translation", "nmt", "translation model", "translate language"],
    "explanation": "Machine translation converts text from a source language to a target language. Neural Machine Translation (NMT) using encoder-decoder transformers replaced statistical phrase-based MT around 2016-2017. Modern MT systems use multilingual encoder-decoder transformers (mBART, NLLB, M2M-100) trained on parallel and back-translated data, supporting hundreds of languages in a single model. BLEU and chrF are the standard metrics."
}
KNOWLEDGE_BASE["text_summarization"] = {
    "aliases": ["text summarization", "summarization", "extractive summarization", "abstractive summarization", "document summarization", "auto summary"],
    "explanation": "Summarization condenses a longer document into a shorter version preserving key information. Extractive summarization selects salient sentences from the source; abstractive summarization generates new text. Modern systems use encoder-decoder transformers (BART, T5, PEGASUS) fine-tuned on (article, summary) pairs from CNN/DailyMail, XSum, etc. ROUGE is the standard metric, though it correlates imperfectly with human judgment."
}
KNOWLEDGE_BASE["question_answering"] = {
    "aliases": ["question answering", "qa", "extractive qa", "open domain qa", "reading comprehension", "squad"],
    "explanation": "Question Answering (QA) systems answer natural language questions. Extractive QA selects an answer span from a given passage (e.g., SQuAD); abstractive QA generates the answer; open-domain QA retrieves relevant documents first. Encoder-only models with span prediction heads handle extractive QA; LLMs handle open-domain QA via RAG or parametric knowledge. Metrics include exact match (EM) and F1."
}
KNOWLEDGE_BASE["natural_language_inference"] = {
    "aliases": ["nli", "natural language inference", "textual entailment", "entailment", "snli", "mnli"],
    "explanation": "Natural Language Inference (NLI) classifies the relationship between a premise and hypothesis as entailment, contradiction, or neutral. SNLI and MNLI are major benchmarks. NLI tests deep semantic understanding and is a core component of GLUE. NLI-trained models are also useful for zero-shot classification: frame any classification problem as 'does this text entail this label description?'"
}
KNOWLEDGE_BASE["dependency_parsing"] = {
    "aliases": ["dependency parsing", "syntactic parsing", "parse tree", "head dependent", "universal dependencies", "constituency parsing"],
    "explanation": "Dependency parsing produces a tree representing grammatical relationships between words, where each word (except the root) has a single head and a labeled dependency relation (subject, object, modifier, etc.). Modern parsers use transition-based or graph-based neural models on top of contextual embeddings. Constituency parsing is an alternative that builds nested phrase structures rather than head-dependent links."
}
KNOWLEDGE_BASE["bleu_score"] = {
    "aliases": ["bleu", "bleu score", "translation metric", "papineni bleu", "ngram precision metric", "evaluate translation"],
    "explanation": "BLEU (Bilingual Evaluation Understudy, Papineni et al., 2002) measures n-gram precision overlap between machine and reference translations, with a brevity penalty to prevent gaming via short outputs. It typically averages 1- to 4-gram precision. BLEU correlates moderately with human judgment at the corpus level but poorly at the sentence level, and it penalizes valid paraphrases. Despite limitations, it remains the standard MT metric."
}
KNOWLEDGE_BASE["rouge_score"] = {
    "aliases": ["rouge", "rouge score", "summarization metric", "rouge n", "rouge l", "lin rouge"],
    "explanation": "ROUGE (Recall-Oriented Understudy for Gisting Evaluation, Lin 2004) is the standard summarization metric. ROUGE-N measures n-gram recall against reference summaries; ROUGE-L uses the longest common subsequence; ROUGE-S uses skip-bigrams. Unlike BLEU's precision focus, ROUGE emphasizes recall — capturing reference content. Like BLEU, it is an imperfect proxy for quality but enables consistent comparison."
}
KNOWLEDGE_BASE["bertscore"] = {
    "aliases": ["bertscore", "bert score", "embedding metric", "contextual similarity metric", "neural metric", "zhang bertscore"],
    "explanation": "BERTScore (Zhang et al., 2020) evaluates generated text by computing cosine similarity between contextual BERT embeddings of candidate and reference tokens, then aligning them greedily. It correlates better with human judgment than n-gram metrics because it captures semantic similarity rather than surface overlap, recognizing valid paraphrases. It is widely used alongside BLEU/ROUGE for translation and summarization evaluation."
}
KNOWLEDGE_BASE["glue_benchmark"] = {
    "aliases": ["glue", "glue benchmark", "superglue", "nlp benchmark", "wang glue", "general language understanding"],
    "explanation": "GLUE (General Language Understanding Evaluation) is a suite of 9 NLP tasks (sentiment, similarity, NLI, etc.) used to evaluate general language understanding. SuperGLUE is its harder successor with reading comprehension and reasoning tasks. GLUE was instrumental in tracking progress from BERT to RoBERTa to T5 to early LLMs. By 2020 models exceeded human baselines, motivating harder benchmarks like BIG-bench, MMLU, and HELM."
}
KNOWLEDGE_BASE["f1_score"] = {
    "aliases": ["f1 score", "f measure", "precision recall", "harmonic mean precision recall", "classification metric", "ner metric"],
    "explanation": "F1 score is the harmonic mean of precision and recall: F1 = 2*P*R/(P+R). It balances both metrics, useful when classes are imbalanced. Macro-F1 averages F1 across classes equally; micro-F1 aggregates all true positives, false positives, and false negatives globally. F1 is the standard metric for NER, QA span prediction, and many classification tasks where accuracy alone is misleading."
}

# =====================================================================
# PART 12 — RETRIEVAL, RAG, AND MODERN INFRASTRUCTURE
# =====================================================================
KNOWLEDGE_BASE["rag"] = {
    "aliases": ["rag", "retrieval augmented generation", "lewis 2020 rag", "retrieval llm", "augment llm with documents", "grounded generation"],
    "explanation": "Retrieval-Augmented Generation (RAG, Lewis et al., 2020) augments an LLM by retrieving relevant documents from an external corpus and conditioning generation on them. A retriever finds the top-k passages for a query; the generator produces an answer using both the query and retrieved context. RAG provides factual grounding, enables citation of sources, and lets the model use information beyond its training cutoff without retraining."
}
KNOWLEDGE_BASE["vector_database"] = {
    "aliases": ["vector database", "vector db", "vector search", "embedding database", "pinecone faiss chroma", "ann search"],
    "explanation": "A vector database stores high-dimensional embeddings and supports efficient nearest-neighbor search using approximate nearest neighbor (ANN) algorithms like HNSW or IVF. Documents are encoded into vectors offline; at query time the query embedding is matched against indexed vectors. Vector databases (FAISS, Pinecone, Weaviate, Chroma, Qdrant) are core infrastructure for RAG and semantic search systems."
}
KNOWLEDGE_BASE["sentence_embeddings"] = {
    "aliases": ["sentence embeddings", "sentence vector", "sbert", "sentence transformers", "reimers gurevych", "semantic search embeddings"],
    "explanation": "Sentence embeddings represent entire sentences or passages as fixed-size vectors where semantically similar texts have nearby vectors. Sentence-BERT (Reimers & Gurevych, 2019) fine-tunes BERT with a siamese network and contrastive loss to produce useful sentence embeddings. Modern alternatives include E5, BGE, and OpenAI's text-embedding-3 series. They power semantic search, clustering, and retrieval for RAG."
}
KNOWLEDGE_BASE["bi_encoder_cross_encoder"] = {
    "aliases": ["bi encoder", "cross encoder", "bi-encoder vs cross-encoder", "siamese vs cross", "retrieval architecture", "rerank cross encoder"],
    "explanation": "Bi-encoders independently encode queries and documents into vectors and compare via dot product or cosine — fast and indexable, used for first-stage retrieval. Cross-encoders concatenate query and document and run a transformer over both, producing a relevance score — much more accurate but cannot be precomputed, so they are used as rerankers over top-k bi-encoder candidates. The two-stage retrieve-then-rerank pipeline is standard."
}
KNOWLEDGE_BASE["bm25"] = {
    "aliases": ["bm25", "okapi bm25", "tf idf bm25", "lexical retrieval", "sparse retrieval", "keyword search"],
    "explanation": "BM25 is a classical lexical ranking function that scores documents based on term frequency, inverse document frequency, and document length normalization. It is the standard sparse retrieval baseline, often outperforming neural methods on out-of-domain data and rare terms. Hybrid retrieval combining BM25 (lexical) with dense embeddings (semantic) typically beats either alone and is widely used in production RAG systems."
}
KNOWLEDGE_BASE["hallucination"] = {
    "aliases": ["hallucination", "llm hallucination", "made up facts", "confabulation llm", "factual errors llm", "ungrounded generation"],
    "explanation": "Hallucination is the generation of plausible-sounding but factually incorrect or fabricated content by LLMs. Causes include training on noisy data, miscalibration of confidence, the next-token-prediction objective rewarding fluency over truth, and lack of grounding. Mitigation strategies include RAG (grounding in retrieved sources), better RLHF, fact-checking pipelines, and explicit uncertainty expression. Hallucination remains a central open problem in LLMs."
}
KNOWLEDGE_BASE["kv_cache"] = {
    "aliases": ["kv cache", "key value cache", "transformer inference cache", "attention cache", "fast inference llm", "autoregressive cache"],
    "explanation": "The KV cache stores the keys and values of all previously generated tokens during autoregressive inference, so they don't need to be recomputed at each new step. Without it, generating token n requires recomputing attention for all n-1 prior tokens. KV cache reduces per-token cost from O(n) to O(1) (amortized). It dominates memory at long contexts, motivating optimizations like multi-query attention, grouped-query attention, and paged attention (vLLM)."
}
KNOWLEDGE_BASE["flash_attention"] = {
    "aliases": ["flash attention", "flashattention", "dao 2022", "memory efficient attention", "tiled attention", "fused attention"],
    "explanation": "FlashAttention (Dao et al., 2022) is an exact (not approximate) attention algorithm that dramatically reduces memory usage and increases speed by tiling the computation and keeping intermediate values in fast SRAM rather than slow HBM. It avoids materializing the full N×N attention matrix in memory, enabling much longer context lengths. FlashAttention-2 and -3 further optimize for newer GPUs and are now standard in LLM training and inference."
}
KNOWLEDGE_BASE["context_window"] = {
    "aliases": ["context window", "context length", "max sequence length", "long context", "context size llm", "input length limit"],
    "explanation": "The context window is the maximum number of tokens an LLM can process at once. Original GPT-3 had 2K; modern models reach 128K (GPT-4 Turbo), 200K (Claude), 1M+ (Gemini, Claude). Extending context is hard because attention is O(n^2) in memory and compute, position encodings degrade beyond training length, and effective use of long context (not just nominal capacity) requires careful training and architectural tricks like RoPE scaling and ring attention."
}
KNOWLEDGE_BASE["scaling_laws"] = {
    "aliases": ["scaling laws", "kaplan scaling laws", "chinchilla", "compute optimal", "model size data size", "hoffmann scaling"],
    "explanation": "Scaling laws (Kaplan et al., 2020; Hoffmann et al., 2022, Chinchilla) describe how language model loss decreases predictably as a power law in parameters, data, and compute. The Chinchilla paper showed earlier large models like GPT-3 were undertrained — for compute-optimal training, parameters and tokens should scale roughly equally. Scaling laws guide training budget allocation and predict the gains from scaling up further."
}
KNOWLEDGE_BASE["emergent_abilities"] = {
    "aliases": ["emergent abilities", "emergent capabilities", "emergence llm", "phase transition llm", "wei emergent", "scale emergent"],
    "explanation": "Emergent abilities are capabilities that appear sharply at a certain model scale and are absent (or near-random) in smaller models. Examples: chain-of-thought reasoning, instruction following, multi-step arithmetic, in-context learning of novel tasks. Whether emergence is genuine or an artifact of evaluation metrics is debated (Schaeffer et al., 2023), but the practical implication is that some behaviors only become useful past a scale threshold."
}
KNOWLEDGE_BASE["multilingual_models"] = {
    "aliases": ["multilingual models", "mbert", "xlm r", "cross lingual", "multilingual nlp", "low resource language"],
    "explanation": "Multilingual models (mBERT, XLM-R, mT5, BLOOM) are trained on text from many languages, learning shared representations that enable cross-lingual transfer: fine-tune on English NER, evaluate on Swahili NER. They face the curse of multilinguality — beyond a certain number of languages, per-language performance degrades unless the model is scaled up. Tokenization is a key challenge; SentencePiece is preferred for its language-agnostic design."
}
KNOWLEDGE_BASE["safety_alignment"] = {
    "aliases": ["ai safety", "alignment", "llm safety", "harmless helpful honest", "constitutional ai", "red teaming"],
    "explanation": "Alignment research aims to make LLMs behave according to human values: helpful, harmless, and honest. Techniques include RLHF, Constitutional AI (training on principles rather than only human labels), red-teaming (adversarial probing for failures), and refusal training. Open challenges include reward hacking, deceptive alignment, jailbreaks, and ensuring models remain aligned as capabilities scale beyond direct human oversight."
}

# ── MODEL INIT ─────────────────────────────────────────────────────────────────
print("🔄 Loading NLP models...")
embedder = SentenceTransformer("all-MiniLM-L6-v2")
vader = SentimentIntensityAnalyzer()

alias_texts = []
alias_labels = []
for concept, data in KNOWLEDGE_BASE.items():
    for alias in data["aliases"]:
        alias_texts.append(alias)
        alias_labels.append(concept)

tfidf = TfidfVectorizer()
tfidf_matrix = tfidf.fit_transform(alias_texts)

for concept, data in KNOWLEDGE_BASE.items():
    data["embedding"] = embedder.encode(data["explanation"], convert_to_tensor=True)

print(f"✅ Pipeline ready — {len(KNOWLEDGE_BASE)} concepts loaded.\n")


# ── NLP FEATURES ───────────────────────────────────────────────────────────────
def detect_intent(question: str):
    q_vec = tfidf.transform([question.lower()])
    scores = cosine_similarity(q_vec, tfidf_matrix).flatten()
    best_idx = int(np.argmax(scores))

    if scores[best_idx] > 0.15:
        return alias_labels[best_idx]

    q_emb = embedder.encode(question, convert_to_tensor=True)
    best_score, best_concept = 0.0, None
    for concept, data in KNOWLEDGE_BASE.items():
        score = float(util.cos_sim(q_emb, data["embedding"]))
        if score > best_score:
            best_score, best_concept = score, concept
    return best_concept if best_score > 0.40 else None


CORRECT_THRESHOLD = 0.72

def evaluate_understanding(user_understanding: str, concept: str) -> dict:
    gt_emb = KNOWLEDGE_BASE[concept]["embedding"]
    usr_emb = embedder.encode(user_understanding, convert_to_tensor=True)
    similarity = float(util.cos_sim(usr_emb, gt_emb))
    return {
        "label": "CORRECT" if similarity >= CORRECT_THRESHOLD else "INCORRECT",
        "similarity": round(similarity, 4),
        "ground_truth": KNOWLEDGE_BASE[concept]["explanation"]
    }


def analyze_sentiment(text: str) -> dict:
    scores = vader.polarity_scores(text)
    compound = scores["compound"]
    if compound >= 0.3:
        tone = "confident"
    elif compound <= -0.2:
        tone = "uncertain"
    else:
        tone = "neutral"
    return {"compound": compound, "tone": tone}


def run_pipeline(question: str, user_understanding: str) -> dict:
    concept = detect_intent(question)
    if not concept:
        return {"label": "OUT_OF_SCOPE", "concept": None,
                "similarity": None, "ground_truth": None, "tone": None}

    similarity_result = evaluate_understanding(user_understanding, concept)
    sentiment_result = analyze_sentiment(user_understanding)

    return {
        "concept": concept,
        "label": similarity_result["label"],
        "similarity": similarity_result["similarity"],
        "ground_truth": similarity_result["ground_truth"],
        "tone": sentiment_result["tone"],
        "compound": sentiment_result["compound"]
    }


# ── SESSION STORE (in-memory, per-session chat history) ────────────────────────
sessions = {}   # session_id -> list of chat messages


# ── ROUTES ─────────────────────────────────────────────────────────────────────
@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "concepts": len(KNOWLEDGE_BASE)})


@app.route("/api/ask", methods=["POST"])
def ask():
    body = request.get_json(force=True)
    question = body.get("question", "").strip()
    understanding = body.get("understanding", "").strip()
    session_id = body.get("session_id", "default")

    if not question or not understanding:
        return jsonify({"error": "Both 'question' and 'understanding' are required."}), 400

    # Run NLP pipeline
    result = run_pipeline(question, understanding)

    if result["label"] == "OUT_OF_SCOPE":
        return jsonify({
            "response": "I only engage with NLP, ML, and AI academic concepts — RNNs, Transformers, LLMs, tokenization, embeddings, RAG, alignment, and more. Ask me something in that space!",
            "analysis": {"concept": None, "label": "OUT_OF_SCOPE", "similarity": None, "tone": None}
        })

    # Build user message with NLP metadata
    user_message = (
        f"Concept/Question: {question}\n"
        f"My understanding: {understanding}\n"
        f"[NLP: label={result['label']}, similarity={result['similarity']}, tone={result['tone']}]"
    )

    # Manage session history
    if session_id not in sessions:
        sessions[session_id] = []
    sessions[session_id].append({"role": "user", "content": user_message})

    # Call Groq LLM
    if not groq_client:
        assistant_reply = (
            f"[GROQ_API_KEY not configured] NLP analysis — Concept: {result['concept']}, "
            f"Label: {result['label']}, Similarity: {result['similarity']}, Tone: {result['tone']}. "
            "Set GROQ_API_KEY in backend/.env to enable LLM responses."
        )
    else:
        try:
            response = groq_client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[{"role": "system", "content": SYSTEM_PROMPT}] + sessions[session_id],
                temperature=0.75,
                max_tokens=250
            )
            assistant_reply = response.choices[0].message.content
        except Exception as e:
            assistant_reply = f"LLM error: {str(e)}"

    sessions[session_id].append({"role": "assistant", "content": assistant_reply})

    return jsonify({
        "response": assistant_reply,
        "analysis": {
            "concept": result["concept"],
            "label": result["label"],
            "similarity": result["similarity"],
            "tone": result["tone"]
        }
    })


@app.route("/api/reset", methods=["POST"])
def reset():
    body = request.get_json(force=True) if request.data else {}
    session_id = body.get("session_id", "default")
    sessions.pop(session_id, None)
    return jsonify({"status": "cleared", "session_id": session_id})


@app.route("/api/concepts", methods=["GET"])
def concepts():
    return jsonify({
        concept: {"aliases": data["aliases"][:3], "preview": data["explanation"][:120] + "..."}
        for concept, data in KNOWLEDGE_BASE.items()
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
