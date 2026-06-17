---
title: English → Indian Languages Machine Translation
emoji: 🌐
colorFrom: green
colorTo: purple
sdk: gradio
sdk_version: 6.12.0
app_file: app.py
pinned: false
---

# 🌐 English → Indian Languages Machine Translation

A complete machine translation pipeline for **English → Indian languages** using the **NLLB-200 multilingual model**, evaluated with multiple automatic metrics and deployed via an interactive **Gradio web application**.

---

# 📋 Table of Contents

- Overview
- Languages Supported
- Dataset
- Models Evaluated
- Evaluation Results
- Cross-Language Evaluation
- Metric Descriptions
- Multilingual Extension Task
- Project Structure
- Quick Start
- Running the App
- Running Evaluation
- Deployment Architecture
- Report

---

# Overview

This project evaluates multiple pretrained multilingual translation models on **English → Tamil** (primary evaluation) and demonstrates multilingual capability across multiple Indian languages using the best performing model.

The best performing model was:

```
facebook/nllb-200-distilled-600M
```

This model supports **200+ languages**, allowing the application to translate English sentences into multiple Indian languages using a single model.

Key observation:

- NLLB-200 significantly outperforms **M2M100** and **T5-Base** for English → Tamil translation.
- The same model can also perform **zero-shot translation** for other Indian languages.

---

# Languages Supported

| Language | Script | NLLB Token | BERTScore Lang |
|--------|--------|------------|---------------|
| Tamil | தமிழ் | `tam_Taml` | ta |
| Hindi | हिन्दी | `hin_Deva` | hi |
| Telugu | తెలుగు | `tel_Telu` | te |
| Kannada | ಕನ್ನಡ | `kan_Knda` | kn |
| Malayalam | മലയാളം | `mal_Mlym` | ml |

Switching languages in the application **does not reload the model**.  
Only the **target language token (`forced_bos_token_id`)** changes.

---

# Dataset

| Property | Value |
|---|---|
| Dataset | ai4bharat/IndicMTEval |
| Primary evaluation language | Tamil |
| Additional demo languages | Hindi, Telugu, Kannada, Malayalam |
| Split used | test |
| Samples per language | up to 200 |

Tamil was selected as the **primary benchmark language** because IndicMTEval provides **human evaluation scores (MQM / Direct Assessment)** for Tamil translations.
### URL:
https://huggingface.co/datasets/ai4bharat/IndicMTEval
---

# Preprocessing Applied

- Lowercasing
- Removing extra whitespace
- Trimming leading and trailing spaces

---

# Models Evaluated

The following translation models were evaluated for **English → Tamil**:

| Model | Parameters | Architecture | Tamil Token |
|------|------------|-------------|------------|
| `facebook/nllb-200-distilled-600M` | 600M | Encoder-Decoder (NLLB) | tam_Taml |
| `facebook/m2m100_418M` | 418M | Encoder-Decoder (M2M100) | ta |
| `t5-base` | 220M | Encoder-Decoder (T5) | prompt-based |

Additional semantic evaluation was performed using:

| Evaluation Model | Purpose |
|---|---|
| `WMT20 COMET-DA` | Neural MT evaluation metric |
| `all-MiniLM-L6-v2` | Sentence embedding similarity |

---

# Evaluation Results

### Primary Model Comparison (English → Tamil)

| Model | BLEU ↑ | chrF ↑ | BERTScore F1 ↑ | Cosine Sim ↑ |
|------|------|------|------|------|
| **NLLB-200 (600M)** 🏆 | **0.142** | **41.3** | **0.618** | **0.731** |
| M2M100 (418M) | 0.098 | 34.7 | 0.581 | 0.694 |
| T5-Base | 0.011 | 12.4 | 0.401 | 0.512 |

NLLB-200 clearly performs best across all evaluation metrics.

---

# Cross-Language Evaluation (Indic MT Benchmark)

To analyze multilingual performance, the **NLLB-200 model** was evaluated across several Indian languages using:

- BLEU
- chrF
- COMET (WMT20 COMET-DA)
- Sentence embedding cosine similarity

Each language used:

```
800 training samples
200 validation samples
```

---

### Validation Results

| Language | BLEU ↑ | chrF ↑ | COMET ↑ | Ensemble Score ↑ |
|------|------|------|------|------|
| Hindi | 31.77 | 37.05 | 0.4404 | 47.20 |
| Tamil | 8.45 | 58.68 | **0.5964** | 49.29 |
| Malayalam | 11.95 | 60.29 | 0.5612 | 50.38 |
| Marathi | **37.04** | **78.40** | 0.4397 | **62.57** |
| Gujarati | 21.03 | 56.23 | 0.5830 | 52.41 |

---

### Best Performers

| Metric | Best Language | Score |
|------|------|------|
| BLEU | Marathi | 37.04 |
| chrF | Marathi | 78.40 |
| COMET | Tamil | **0.5964** |
| Ensemble Score | Marathi | **62.57** |

---

### Evaluation Insights

- Marathi shows strong lexical alignment with the reference translations.
- Tamil achieves the **highest COMET score**, indicating strong semantic alignment.
- Overall multilingual performance is stable across languages.

```
Average Ensemble Score: 52.37 / 100
Best Ensemble Score: 62.57 (Marathi)
Worst Ensemble Score: 47.20 (Hindi)
Score Spread: 15.37 points
```

All experiments were performed using the **pretrained NLLB-200 model in a zero-shot translation setting**.

---

# Metric Descriptions

BLEU  
Measures n-gram overlap between predicted translation and reference translation.

chrF  
Character-level F-score suited for morphologically rich languages.

BERTScore  
Uses contextual embeddings to measure semantic similarity.

COMET  
A neural metric trained to predict human translation quality judgments.

Cosine Similarity  
Measures similarity between sentence embeddings of predicted and reference translations.

---

# Multilingual Extension Task

Since the NLLB model supports **200+ languages**, the application was extended to support translation into multiple Indian languages using the same model.

Supported translations:

- English → Tamil
- English → Hindi
- English → Telugu
- English → Kannada
- English → Malayalam

Instead of loading multiple models, the system simply changes the **target language token**:

```
model.generate(
    **inputs,
    forced_bos_token_id=tokenizer.convert_tokens_to_ids("tam_Taml")
)
```

This allows multilingual translation using a **single model instance**.

---

# Project Structure

```
en-tamil-mt/
├── app/
│   └── app.py
├── evaluation/
│   ├── evaluate_models.py
│   └── evaluate_multilingual.py
├── notebooks/
│   └── evaluation.ipynb
├── docs/
│   └── report.md
├── requirements.txt
├── .gitignore
└── README.md
```

---

# Quick Start

Clone the repository:

```
git clone https://github.com/<your-username>/en-tamil-mt.git
cd en-tamil-mt
```

Create environment:

```
python -m venv venv
source venv/bin/activate
```

Install dependencies:

```
pip install -r requirements.txt
```

---

# Running the App

Run locally:

```
python app/app.py
```

## Run Locally

Open:

http://localhost:7860

## Live Demo

https://huggingface.co/spaces/ashrithagowthami/Machine-Translation
# Running Evaluation

Evaluate model comparison:

```
python evaluation/evaluate_models.py --model all --samples 200
```

Multilingual evaluation:

```
python evaluation/evaluate_multilingual.py --lang all --samples 200
```

---

# Deployment Architecture

```
User Input (English)
        │
        ▼
Gradio UI
        │
        ▼
Preprocessing
        │
        ▼
NLLB Tokenizer
        │
        ▼
NLLB Model.generate()
        │
        ▼
Target language token
        │
        ▼
Translated output
```

---

# Report

Full project report available in:

```
docs/report.md
```

The report includes:

- Dataset analysis
- Model comparison
- Evaluation methodology
- System architecture
- Results and conclusions

# Contributor 
- Ashritha Gowthami Nelakurthi   
