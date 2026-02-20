# 🎬 BERT Sentiment Analysis

<p align="center">
  <img src="https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white"/>
  <img src="https://img.shields.io/badge/HuggingFace-FFD21E?style=for-the-badge&logo=huggingface&logoColor=black"/>
  <img src="https://img.shields.io/badge/FastAPI-009688?style=for-the-badge&logo=fastapi&logoColor=white"/>
  <img src="https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white"/>
  <img src="https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikitlearn&logoColor=white"/>
</p>

<p align="center">
  <strong>Fine-tuned BERT model for binary sentiment classification, served via a production-ready REST API.</strong><br/>
  Trained end-to-end on the IMDB dataset — <em>on a CPU</em> 
</p>

---

## 📖 Table of Contents

- [What This Does](#-what-this-does)
- [Model Architecture](#-model-architecture)
- [Project Structure](#-project-structure)
- [How the Dataset is Processed](#️-how-the-dataset-is-processed)
- [How Training Works](#️-how-training-works)
- [Configuration](#️-configuration)
- [Getting Started](#-getting-started)
- [API Reference](#-api-reference)
- [Optimization Tips](#-optimization-tips)
- [Known Limitations](#-known-limitations)
- [Built With](#️-built-with)

---

## ✨ What This Does

This project fine-tunes Google's **BERT (Bidirectional Encoder Representations from Transformers)** on the IMDB movie review dataset to classify text as either **positive** or **negative** sentiment.

Unlike traditional NLP approaches (TF-IDF + Logistic Regression, LSTMs, etc.), BERT reads text **bidirectionally** — it understands context from both the left and right of every token simultaneously. This makes it far better at capturing nuanced language like sarcasm, negation, and complex sentence structure.

Given any piece of text, the model returns a confidence score for both classes in under 250ms, served through a clean FastAPI REST endpoint.

```json
POST /predict
{
  "sentence": "This movie was absolutely amazing!"
}

→ {
  "positive": 0.973,
  "negative": 0.027,
  "sentence": "This movie was absolutely amazing!",
  "time_taken": 0.2279
}
```

> The `positive` and `negative` scores always sum to `1.0`. They are derived by applying sigmoid to the raw logit output to get `positive`, then computing `negative = 1 - positive`.

---

## 🧠 Model Architecture

```
Input Text
    ↓
BertTokenizer (bert-base-uncased)
  • WordPiece tokenization
  • Lowercases all text (uncased model)
  • Adds [CLS] token at start, [SEP] token at end
  • Pads / truncates to MAX_LEN = 32 tokens
  • Returns: input_ids, attention_mask, token_type_ids
    ↓
BERT Base Uncased  (bert-base-uncased)
  • 12 Transformer encoder layers
  • 768 hidden dimensions
  • 12 self-attention heads
  • ~110M parameters
  • Pretrained on BooksCorpus + English Wikipedia
    ↓
[CLS] Pooler Output  →  dense 768-dim sentence vector
    ↓
Dropout(p=0.3)         ← prevents overfitting on small fine-tune data
    ↓
Linear(768 → 1)        ← learnable classification head
    ↓
BCEWithLogitsLoss      ← fused sigmoid + BCE loss during training
Sigmoid                ← applied separately at inference only
    ↓
Sentiment Score ∈ [0, 1]
```

### Why `BCEWithLogitsLoss` and not plain `BCELoss`?

PyTorch's `BCEWithLogitsLoss` fuses the sigmoid and binary cross-entropy into a single numerically stable operation using the **log-sum-exp trick**. This prevents floating point overflow when logits are very large or very negative. Because of this, the model's `forward()` returns raw logits — sigmoid is only applied separately in `app.py` at inference time, never during training.

### Why use the `[CLS]` token's pooler output?

BERT prepends a special `[CLS]` (classification) token to every input. The `pooler_output` is a 768-dim dense representation derived from this token, passed through a tanh activation inside BERT. It acts as an **aggregate representation of the entire input sequence** and is the standard choice for sentence-level classification tasks. It's also what BERT was pretrained with for NSP (Next Sentence Prediction), making it naturally suited for downstream binary classification.

### Why Dropout before the classification head?

BERT has 110M parameters but we're fine-tuning on only a few thousand examples. Dropout (p=0.3) randomly zeros 30% of the pooler activations during training, forcing the network to learn redundant representations and significantly reducing overfitting on the small downstream dataset.

### What are `token_type_ids`?

BERT supports two-sentence inputs (e.g., for question answering or NLI), using `token_type_ids` to distinguish sentence A (all `0`s) from sentence B (all `1`s). For single-sentence tasks like ours, they're all `0` — but still required by the model's forward signature, which is why they're passed through the full pipeline from dataset to model.

### What does `attention_mask` do?

Since all sequences are padded to `MAX_LEN`, the `attention_mask` tells BERT which tokens are real (`1`) and which are padding (`0`). This prevents the self-attention mechanism from attending to meaningless padding positions, keeping the sentence representations clean and accurate.

---

## 📁 Project Structure

```
├── config.py        # Central config: hyperparameters, paths, tokenizer
├── dataset.py       # PyTorch Dataset — tokenization and tensor creation
├── model.py         # BERT model with dropout + linear classification head
├── engine.py        # Training loop (train_fn) and evaluation loop (eval_fn)
├── train.py         # Full training pipeline with checkpointing
├── app.py           # FastAPI inference server with /predict endpoint
├── model.bin        # Saved model weights (generated after training)
└── input/
    └── IMDB Dataset.csv    # Raw training data (50K reviews)
```

Each file has a single, well-defined responsibility. `config.py` acts as the **single source of truth** for all hyperparameters — changing a value there propagates everywhere automatically without hunting through multiple files.

---

## 🗂️ How the Dataset is Processed

**File:** `dataset.py`

The `BERTDataset` class wraps the IMDB reviews into a standard PyTorch `Dataset`. For each review it:

1. **Normalizes text** — strips extra whitespace with `" ".join(review.split())`
2. **Tokenizes** — uses `BertTokenizer.encode_plus()` which handles everything in one call: adding special tokens, padding to `MAX_LEN`, truncating long inputs, and returning all required tensor dictionaries
3. **Creates tensors** — `input_ids`, `attention_mask`, and `token_type_ids` are each cast to `torch.long`; targets to `torch.float` (required by `BCEWithLogitsLoss` which expects float targets)

In `train.py`, the raw CSV is preprocessed as follows:

- **5,000 rows are randomly sampled** (with `random_state=42` for reproducibility) to keep CPU training time manageable from the full 50K dataset
- `sentiment` string labels are mapped: `"positive" → 1`, `"negative" → 0`
- A **stratified 90/10 train/validation split** is performed via `train_test_split(..., stratify=dfx.sentiment.values)`, ensuring both splits maintain the original class balance — important for getting reliable validation metrics

---

## 🏋️ How Training Works

**File:** `train.py` + `engine.py`

### Optimizer: AdamW with Differential Weight Decay

Not all parameters should be regularized equally. The training script splits the model's parameters into two groups:

```python
no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]

# Group 1: All weights (Linear, attention projections) → weight_decay = 0.001
# Group 2: Biases + LayerNorm parameters               → weight_decay = 0.0
```

Applying L2 weight decay to `LayerNorm` weights and bias terms is known to harm training stability and is explicitly excluded. This matches the original BERT fine-tuning recipe from the Google paper.

### Learning Rate Scheduler

A **linear decay schedule** (`get_linear_schedule_with_warmup`) decays the learning rate from `3e-5` linearly down to `0` over the full training run:

```
num_train_steps = (num_samples / batch_size) * num_epochs
```

This prevents **catastrophic forgetting** — a high, constant learning rate when fine-tuning would overwrite the rich representations BERT learned during pretraining. The small LR + decay ensures the pretrained weights are gently nudged rather than overwritten.

### Training Loop (`engine.py` — `train_fn`)

```
For each batch:
  1. Move ids, mask, token_type_ids, targets to device
  2. Zero gradients
  3. Forward pass → raw logits (shape: [batch_size, 1])
  4. Compute BCEWithLogitsLoss(logits, targets.view(-1, 1))
  5. loss.backward() — compute gradients
  6. optimizer.step() — update weights
  7. scheduler.step() — decay learning rate
```

### Evaluation Loop (`engine.py` — `eval_fn`)

```
Under torch.no_grad() context (no gradient computation):
  For each batch:
    1. Forward pass → raw logits
    2. Apply sigmoid → probabilities in [0, 1]
    3. Accumulate all outputs and targets as Python lists
  Return: complete outputs + targets for metric computation
```

### Best Model Checkpointing

After each epoch, validation probabilities are thresholded at `0.5` to produce binary predictions, and `accuracy_score` is computed against ground truth. **The checkpoint is only saved when the current epoch beats the previous best accuracy.** This ensures `model.bin` always holds the best-performing weights, not necessarily the last epoch's weights.

---

## ⚙️ Configuration

All hyperparameters are centralized in `config.py`:

| Parameter | Value | Description |
|---|---|---|
| `MAX_LEN` | `32` | Max token sequence length (truncation/padding target) |
| `TRAIN_BATCH_SIZE` | `8` | Batch size during training |
| `VALID_BATCH_SIZE` | `4` | Batch size during validation |
| `EPOCHS` | `2` | Number of training epochs |
| `BERT_PATH` | `bert-base-uncased` | HuggingFace model identifier |
| `MODEL_PATH` | `model.bin` | Path to save/load the best checkpoint |
| `TRAINING_FILE` | `input/IMDB Dataset.csv` | Path to training CSV |
| `DEVICE` | `cpu` | Compute device (`"cuda"` for GPU) |

The `TOKENIZER` is also instantiated here once and reused across `dataset.py` and `app.py` — avoiding redundant model loading.

---

## 🚀 Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/bert-sentiment-analysis.git
cd bert-sentiment-analysis
```

### 2. Install Dependencies

```bash
pip install torch transformers fastapi uvicorn scikit-learn pandas pydantic tqdm
```

### 3. Download the Dataset

Get the [IMDB Dataset from Kaggle](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews) and place it at:

```
input/IMDB Dataset.csv
```

The CSV must have two columns: `review` (text) and `sentiment` (`"positive"` / `"negative"`).

### 4. Train the Model

```bash
python train.py
```

This will sample 5,000 reviews, train for 2 epochs with validation after each, and save the best checkpoint to `model.bin`. Expect **45–90 minutes on a modern CPU**.

### 5. Launch the Inference API

```bash
uvicorn app:app --reload --host 0.0.0.0 --port 8000
```

### 6. Test a Prediction

**cURL:**
```bash
curl -X POST "http://127.0.0.1:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{"sentence": "One of the best films I have ever seen!"}'
```

**Python:**
```python
import requests

response = requests.post(
    "http://127.0.0.1:8000/predict",
    json={"sentence": "The plot was terrible and the acting was worse."}
)
print(response.json())
```

**Swagger UI:**
Visit `http://127.0.0.1:8000/docs` for an interactive API explorer, auto-generated by FastAPI.

---

## 🌐 API Reference

### `GET /`

Health check endpoint.

**Response:**
```json
{ "message": "Sentiment API is running" }
```

---

### `POST /predict`

Run sentiment inference on a piece of text.

**Request Body:**

| Field | Type | Description |
|---|---|---|
| `sentence` | `string` | The text to classify |

**Response:**

| Field | Type | Description |
|---|---|---|
| `positive` | `float` | Confidence the text is positive (0–1) |
| `negative` | `float` | Confidence the text is negative (0–1) |
| `sentence` | `string` | The original input text |
| `time_taken` | `float` | End-to-end inference time in seconds |

> `positive + negative` always equals `1.0`

**Example Response:**
```json
{
  "positive": 0.9730021953582764,
  "negative": 0.026997804641723633,
  "sentence": "This movie was absolutely amazing!",
  "time_taken": 0.2279
}
```

---

## 🔧 Optimization Tips

### Swap in DistilBERT for ~2x Speed

DistilBERT is a distilled version of BERT — 40% smaller, 60% faster, retaining ~97% of BERT's performance. Update `config.py`:

```python
BERT_PATH = "distilbert-base-uncased"
```

Then update `model.py` to use `DistilBertModel` — note that DistilBERT does not use `token_type_ids` and uses `last_hidden_state` instead of `pooler_output`:

```python
class BERTBaseUncased(nn.Module):
    def __init__(self):
        super().__init__()
        self.bert = transformers.DistilBertModel.from_pretrained(config.BERT_PATH)
        self.bert_drop = nn.Dropout(0.3)
        self.out = nn.Linear(768, 1)

    def forward(self, ids, mask, token_type_ids=None):
        output = self.bert(input_ids=ids, attention_mask=mask)
        pooled = output.last_hidden_state[:, 0]  # [CLS] token representation
        return self.out(self.bert_drop(pooled))
```

### Increase `MAX_LEN` for Better Accuracy

`MAX_LEN = 32` is aggressive — most IMDB reviews are much longer, so significant context is being truncated right now. Increasing to `128` or `256` will improve accuracy, at the cost of memory and training speed.

### Train on the Full 50K Dataset

The script currently samples only 5,000 of 50,000 available reviews. Remove the `.sample(5000)` line in `train.py` to use the entire dataset. Expect ~10x longer training time.

### Enable GPU Training

One line change in `config.py`:

```python
DEVICE = "cuda"   # NVIDIA GPU
DEVICE = "mps"    # Apple Silicon (M1/M2/M3)
```

### Add Batch Inference to the API

For high-throughput use cases, the `/predict` endpoint can be extended to accept a list of sentences and process them in a single forward pass, dramatically improving throughput.

---

## ⚠️ Known Limitations

- **Aggressive truncation at `MAX_LEN = 32`** — Most IMDB reviews are hundreds of tokens long. Setting `MAX_LEN = 32` means the model only ever sees the first ~25 real words of any review. BERT supports up to 512 tokens.
- **Binary classification only** — The model outputs a single positive/negative score. It cannot detect mixed sentiment, neutrality, or sentiment intensity.
- **Domain shift** — Trained exclusively on movie reviews. Accuracy may degrade on other domains (product reviews, social media, financial news) without domain-specific fine-tuning.
- **No batching in the API** — The `/predict` endpoint processes one sentence at a time. For production use, implement batched inference.
- **CPU inference latency** — ~230ms per request on CPU. A GPU would reduce this to single-digit milliseconds.

---

## 🛠️ Built With

| Library | Role |
|---|---|
| [PyTorch](https://pytorch.org/) | Model definition, training loop, inference |
| [HuggingFace Transformers](https://huggingface.co/docs/transformers) | BERT architecture, pretrained weights & tokenizer |
| [FastAPI](https://fastapi.tiangolo.com/) | REST API server with auto-generated Swagger docs |
| [scikit-learn](https://scikit-learn.org/) | Stratified train/val split, accuracy metric |
| [pandas](https://pandas.pydata.org/) | Dataset loading and label preprocessing |
| [Pydantic](https://docs.pydantic.dev/) | API request/response schema validation |
| [tqdm](https://github.com/tqdm/tqdm) | Training progress bars |

---
