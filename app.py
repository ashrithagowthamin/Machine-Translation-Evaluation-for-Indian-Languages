"""
English → Indian Languages Machine Translation App
Model : facebook/nllb-200-distilled-600M  (Best Performing)
UI    : Gradio with custom dark-editorial theme
Languages: Tamil · Hindi · Telugu · Kannada · Malayalam . Gujarathi
"""

import gradio as gr
from transformers import NllbTokenizer, AutoModelForSeq2SeqLM
import torch
import re
import time

# ─────────────────────────────────────────────────────────────────────────────
# Language Configuration
# ─────────────────────────────────────────────────────────────────────────────

# Each entry: NLLB token | display name | native script name | BERTScore lang | evaluation metrics
LANGUAGES = {
    "Tamil": {
        "token":     "tam_Taml",
        "native":    "தமிழ்",
        "flag":      "🇮🇳",
        "bert_lang": "ta",
        "metrics":   {"bleu": 0.142, "chrf": 41.3, "bert": 0.618, "cosine": 0.731},
    },
    "Hindi": {
        "token":     "hin_Deva",
        "native":    "हिन्दी",
        "flag":      "🇮🇳",
        "bert_lang": "hi",
        "metrics":   {"bleu": 0.213, "chrf": 48.7, "bert": 0.671, "cosine": 0.768},
    },
    "Telugu": {
        "token":     "tel_Telu",
        "native":    "తెలుగు",
        "flag":      "🇮🇳",
        "bert_lang": "te",
        "metrics":   {"bleu": 0.138, "chrf": 39.4, "bert": 0.604, "cosine": 0.718},
    },
    "Kannada": {
        "token":     "kan_Knda",
        "native":    "ಕನ್ನಡ",
        "flag":      "🇮🇳",
        "bert_lang": "kn",
        "metrics":   {"bleu": 0.127, "chrf": 37.8, "bert": 0.597, "cosine": 0.709},
    },
    "Malayalam": {
        "token":     "mal_Mlym",
        "native":    "മലയാളം",
        "flag":      "🇮🇳",
        "bert_lang": "ml",
        "metrics":   {"bleu": 0.131, "chrf": 38.6, "bert": 0.601, "cosine": 0.714},
    },
}

LANGUAGE_CHOICES = list(LANGUAGES.keys())

# ─────────────────────────────────────────────────────────────────────────────
# Model Loading  (one NLLB model handles all languages — no reload needed)
# ─────────────────────────────────────────────────────────────────────────────
MODEL_NAME = "facebook/nllb-200-distilled-600M"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Loading model: {MODEL_NAME} on {DEVICE}...")
tokenizer = NllbTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME).to(DEVICE)
model.eval()
print("Model ready ✓")


# ─────────────────────────────────────────────────────────────────────────────
# Preprocessing
# ─────────────────────────────────────────────────────────────────────────────
def preprocess(text: str) -> str:
    text = text.lower()
    text = re.sub(r"\s+", " ", text)
    return text.strip()


# ─────────────────────────────────────────────────────────────────────────────
# Translation  (language-aware via forced_bos_token_id)
# ─────────────────────────────────────────────────────────────────────────────
def translate(text: str, target_language: str, num_beams: int = 4, max_length: int = 256):
    if not text.strip():
        return "", "⚠️  Please enter some text to translate.", metrics_html(target_language)

    lang_cfg = LANGUAGES[target_language]
    nllb_token = lang_cfg["token"]

    start = time.time()
    clean = preprocess(text)

    inputs = tokenizer(
        clean,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=512,
    ).to(DEVICE)

    with torch.no_grad():
        generated = model.generate(
            **inputs,
            forced_bos_token_id=tokenizer.convert_tokens_to_ids(nllb_token),
            num_beams=num_beams,
            max_length=max_length,
            early_stopping=True,
        )

    result = tokenizer.decode(generated[0], skip_special_tokens=True)
    elapsed = time.time() - start

    status = (
        f"✅  {len(text.split())} words → {target_language} ({nllb_token})  "
        f"|  {elapsed:.2f}s  |  {num_beams} beams  |  {DEVICE.upper()}"
    )
    return result, status, metrics_html(target_language)


# ─────────────────────────────────────────────────────────────────────────────
# Dynamic metric cards  (swaps when user changes language dropdown)
# ─────────────────────────────────────────────────────────────────────────────
def metrics_html(language: str) -> str:
    m = LANGUAGES[language]["metrics"]
    native = LANGUAGES[language]["native"]
    return f"""
    <div id="metrics">
        <div class="metric-card"><span class="val">{m['bleu']:.3f}</span><span class="lbl">BLEU</span></div>
        <div class="metric-card"><span class="val">{m['chrf']:.1f}</span><span class="lbl">chrF</span></div>
        <div class="metric-card"><span class="val">{m['bert']:.3f}</span><span class="lbl">BERTScore F1</span></div>
        <div class="metric-card"><span class="val">{m['cosine']:.3f}</span><span class="lbl">Cosine Sim</span></div>
    </div>
    <p style="text-align:center;color:var(--muted);font-size:0.78rem;margin:0 0 24px;">
        Evaluation metrics for English → {language} ({native}) on IndicMTEval · NLLB-200
    </p>
    """


def on_language_change(language: str):
    """Update output textbox label + metric cards when dropdown changes."""
    cfg = LANGUAGES[language]
    new_label = f"{cfg['flag']} {language} Translation  ·  {cfg['native']}"
    return gr.update(label=new_label), metrics_html(language)


# ─────────────────────────────────────────────────────────────────────────────
# Example Sentences
# ─────────────────────────────────────────────────────────────────────────────
EXAMPLES = [
    ["The sun rises in the east and sets in the west.",                "Tamil"],
    ["Artificial intelligence is reshaping how we live and work.",     "Hindi"],
    ["She went to the market to buy fresh vegetables and fruits.",     "Telugu"],
    ["The children played happily in the park after school.",          "Kannada"],
    ["Please book a train ticket from Chennai to Coimbatore.",        "Tamil"],
    ["Climate change is one of the most pressing global challenges.", "Malayalam"],
]


# ─────────────────────────────────────────────────────────────────────────────
# CSS  — dark editorial, India saffron + deep navy palette
# ─────────────────────────────────────────────────────────────────────────────
CSS = """
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:ital,wght@0,600;0,700;1,600&family=DM+Sans:wght@300;400;500&display=swap');

:root {
    --saffron:   #FF6B35;
    --deep-navy: #0D1B2A;
    --slate:     #1C2E40;
    --card:      #162032;
    --border:    #263A50;
    --text:      #E8EDF2;
    --muted:     #7A94AA;
    --gold:      #E5A020;
    --teal:      #2EC4B6;
    --green:     #3DD68C;
    --radius:    12px;
}

body, .gradio-container { background:var(--deep-navy) !important; font-family:'DM Sans',sans-serif !important; color:var(--text) !important; }

/* ── Header ── */
#header { text-align:center; padding:48px 0 28px; border-bottom:1px solid var(--border); margin-bottom:24px; }
#header h1 { font-family:'Playfair Display',serif !important; font-size:2.5rem !important; font-weight:700 !important; color:var(--text) !important; letter-spacing:-0.5px; margin:0 0 6px; }
#header h1 span { color:var(--saffron); }
#header p { color:var(--muted); font-size:0.92rem; font-weight:300; margin:0; }

/* ── Language selector ── */
#lang-selector-row { display:flex; align-items:center; justify-content:center; gap:14px; margin-bottom:18px; flex-wrap:wrap; }
.lang-src { background:var(--slate); border:1px solid var(--border); border-radius:999px; padding:7px 20px; font-size:0.88rem; font-weight:500; color:var(--text); }
.arrow-icon { color:var(--saffron); font-size:1.3rem; font-weight:700; }

/* ── Textboxes ── */
.input-box textarea, .output-box textarea {
    background:var(--card) !important; border:1px solid var(--border) !important;
    border-radius:var(--radius) !important; color:var(--text) !important;
    font-family:'DM Sans',sans-serif !important; font-size:1rem !important;
    line-height:1.7 !important; padding:16px !important; resize:vertical !important;
    transition:border-color 0.2s;
}
.input-box textarea:focus { border-color:var(--saffron) !important; outline:none !important; box-shadow:0 0 0 3px rgba(255,107,53,0.12) !important; }
.output-box textarea { border-color:var(--teal) !important; background:rgba(46,196,182,0.04) !important; }
label span { color:var(--muted) !important; font-size:0.75rem !important; font-weight:500 !important; letter-spacing:0.09em !important; text-transform:uppercase !important; }

/* ── Translate button ── */
#translate-btn {
    background:linear-gradient(135deg, var(--saffron), #C94010) !important;
    color:white !important; border:none !important; border-radius:var(--radius) !important;
    font-family:'DM Sans',sans-serif !important; font-size:1rem !important; font-weight:500 !important;
    padding:14px 0 !important; width:100% !important; cursor:pointer !important;
    transition:opacity 0.2s,transform 0.1s !important; letter-spacing:0.02em;
}
#translate-btn:hover  { opacity:0.88 !important; transform:translateY(-1px) !important; }
#translate-btn:active { transform:translateY(0) !important; }

/* ── Status bar ── */
#status-box textarea {
    background:var(--slate) !important; border:1px solid var(--border) !important;
    border-radius:8px !important; color:var(--green) !important; font-size:0.78rem !important;
    font-family:monospace !important; padding:10px 14px !important; min-height:unset !important; resize:none !important;
}

/* ── Accordion / sliders ── */
.gr-accordion { background:var(--card) !important; border:1px solid var(--border) !important; border-radius:var(--radius) !important; }
input[type=range] { accent-color:var(--saffron) !important; }

/* ── Metric cards ── */
#metrics { display:grid; grid-template-columns:repeat(4,1fr); gap:12px; margin:18px 0 6px; }
.metric-card { background:var(--card); border:1px solid var(--border); border-radius:var(--radius); padding:18px; text-align:center; }
.metric-card .val { font-family:'Playfair Display',serif; font-size:1.55rem; color:var(--gold); display:block; }
.metric-card .lbl { font-size:0.7rem; color:var(--muted); text-transform:uppercase; letter-spacing:0.1em; margin-top:4px; display:block; }

/* ── Examples ── */
.gr-examples table { background:var(--card) !important; border-radius:var(--radius) !important; }
.gr-examples td { color:var(--muted) !important; border-color:var(--border) !important; font-size:0.86rem !important; }
.gr-examples tr:hover td { color:var(--text) !important; }

/* ── Footer ── */
#footer { text-align:center; padding:24px 0 10px; border-top:1px solid var(--border); margin-top:36px; color:var(--muted); font-size:0.78rem; line-height:1.9; }
"""


# ─────────────────────────────────────────────────────────────────────────────
# Build Gradio UI
# ─────────────────────────────────────────────────────────────────────────────
with gr.Blocks(css=CSS, title="English → Indian Languages Translator") as demo:

    # Header
    gr.HTML("""
    <div id="header">
        <h1>English → <span>Indian Languages</span></h1>
        <p>Neural machine translation powered by NLLB-200 · Tamil · Hindi · Telugu · Kannada · Malayalam</p>
    </div>
    """)

    # Language selector row
    with gr.Row(elem_id="lang-selector-row"):
        gr.HTML('<span class="lang-src">🇬🇧 English</span><span class="arrow-icon">→</span>')
        lang_dropdown = gr.Dropdown(
            choices=LANGUAGE_CHOICES,
            value="Tamil",
            label="",
            show_label=False,
            scale=0,
            min_width=200,
        )

    # Dynamic metric cards
    metric_display = gr.HTML(metrics_html("Tamil"))

    # Translation panel
    with gr.Row():
        with gr.Column(scale=1):
            src_text = gr.Textbox(
                label="English Source",
                placeholder="Type or paste English text here…",
                lines=8,
                elem_classes=["input-box"],
            )
        with gr.Column(scale=1):
            tgt_text = gr.Textbox(
                label="🇮🇳 Tamil Translation  ·  தமிழ்",
                lines=8,
                interactive=False,
                elem_classes=["output-box"],
            )

    translate_btn = gr.Button("⟶  Translate", elem_id="translate-btn")
    status_box = gr.Textbox(label="", interactive=False, lines=1, elem_id="status-box")

    # Advanced settings
    with gr.Accordion("⚙️  Advanced Settings", open=False):
        with gr.Row():
            num_beams  = gr.Slider(minimum=1, maximum=8, value=4, step=1,
                                   label="Beam Width  (higher = better quality, slower)")
            max_length = gr.Slider(minimum=64, maximum=512, value=256, step=32,
                                   label="Max Output Tokens")

    # Examples
    gr.Examples(
        examples=EXAMPLES,
        inputs=[src_text, lang_dropdown],
        label="📌 Try an Example",
    )

    # Footer
    gr.HTML("""
    <div id="footer">
        Model: facebook/nllb-200-distilled-600M &nbsp;·&nbsp; Dataset: ai4bharat/IndicMTEval<br>
        NLLB tokens: tam_Taml · hin_Deva · tel_Telu · kan_Knda · mal_Mlym
    </div>
    """)

    # ── Event wiring ─────────────────────────────────────────────────────────

    translate_btn.click(
        fn=translate,
        inputs=[src_text, lang_dropdown, num_beams, max_length],
        outputs=[tgt_text, status_box, metric_display],
    )
    src_text.submit(
        fn=translate,
        inputs=[src_text, lang_dropdown, num_beams, max_length],
        outputs=[tgt_text, status_box, metric_display],
    )
    lang_dropdown.change(
        fn=on_language_change,
        inputs=[lang_dropdown],
        outputs=[tgt_text, metric_display],
    )


if __name__ == "__main__":
    demo.launch(share=False, server_name="0.0.0.0", server_port=7860)
