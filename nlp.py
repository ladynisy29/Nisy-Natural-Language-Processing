import nltk
#nltk.download('punkt')
import gradio as gr
import nltk
import torch
import torch.nn.functional as F
from nltk.tokenize import word_tokenize
from transformers import (
    pipeline,
    AutoTokenizer,
    AutoModelForSequenceClassification
)
from sentence_transformers import SentenceTransformer

# ===============================
# Load Models (Load once globally)
# ===============================

# Tokenizer comparison (Hugging Face BPE)
hf_tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

# Sentiment Analysis
sentiment_pipeline = pipeline(
    "sentiment-analysis",
    model="distilbert-base-uncased-finetuned-sst-2-english"
)

# Semantic Similarity Model
similarity_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# Zero-Shot Classification
zero_shot_pipeline = pipeline(
    "zero-shot-classification",
    model="facebook/bart-large-mnli"
)

# Summarization (Innovation Tab)
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

# ===============================
# TAB 1: Tokenization
# ===============================

def tokenize_text(text):
    # NLTK Word-level
    nltk_tokens = word_tokenize(text)

    # Hugging Face Subword (BPE)
    hf_tokens = hf_tokenizer.tokenize(text)

    return (
        str(nltk_tokens),
        len(nltk_tokens),
        str(hf_tokens),
        len(hf_tokens),
    )

# ===============================
# TAB 2: Sentiment Analysis
# ===============================

def analyze_sentiment(text):
    result = sentiment_pipeline(text)[0]
    label = result["label"]
    score = result["score"]

    if label == "POSITIVE":
        return f"Positive: {score*100:.2f}%\nNegative: {(1-score)*100:.2f}%"
    else:
        return f"Positive: {(1-score)*100:.2f}%\nNegative: {score*100:.2f}%"

# ===============================
# TAB 3: Semantic Similarity
# ===============================

def compute_similarity(text1, text2):
    embeddings = similarity_model.encode([text1, text2], convert_to_tensor=True)
    similarity = F.cosine_similarity(embeddings[0], embeddings[1], dim=0)
    return f"Cosine Similarity Score: {similarity.item():.4f}"

# ===============================
# TAB 4: Zero-Shot Classification
# ===============================

def zero_shot_classify(text, labels):
    label_list = [l.strip() for l in labels.split(",")]
    result = zero_shot_pipeline(text, label_list)

    output = ""
    for label, score in zip(result["labels"], result["scores"]):
        output += f"{label}: {score*100:.2f}%\n"

    return output

# ===============================
# TAB 5: Summarization (Innovation)
# ===============================

def summarize_text(text):
    summary = summarizer(text, max_length=100, min_length=30, do_sample=False)
    return summary[0]["summary_text"]

# ===============================
# Gradio UI
# ===============================

with gr.Blocks(title="Integrated NLP Toolkit") as demo:

    gr.Markdown("# 🧠 Integrated NLP Toolkit")
    gr.Markdown("Explore NLP from tokenization to semantic understanding.")

    with gr.Tabs():

        # ---------------- TAB 1 ----------------
        with gr.Tab("Text Preprocessing (Tokenization)"):

            input_text = gr.Textbox(
                value="NLP is transforming the way machines understand human language.",
                label="Enter Text",
                lines=3
            )

            btn1 = gr.Button("Tokenize")

            nltk_output = gr.Textbox(label="NLTK Word Tokens")
            nltk_count = gr.Number(label="NLTK Token Count")

            hf_output = gr.Textbox(label="Hugging Face BPE Tokens")
            hf_count = gr.Number(label="HF Token Count")

            btn1.click(
                tokenize_text,
                inputs=input_text,
                outputs=[nltk_output, nltk_count, hf_output, hf_count]
            )

        # ---------------- TAB 2 ----------------
        with gr.Tab("Sentiment Analysis"):

            sentiment_input = gr.Textbox(
                value="I absolutely love how powerful modern AI models are!",
                label="Enter Text",
                lines=3
            )

            sentiment_btn = gr.Button("Analyze Sentiment")
            sentiment_output = gr.Textbox(label="Sentiment Breakdown")

            sentiment_btn.click(
                analyze_sentiment,
                inputs=sentiment_input,
                outputs=sentiment_output
            )

        # ---------------- TAB 3 ----------------
        with gr.Tab("Semantic Similarity"):

            text1 = gr.Textbox(
                value="Artificial Intelligence is revolutionizing technology.",
                label="Text 1"
            )

            text2 = gr.Textbox(
                value="AI is changing the future of tech industries.",
                label="Text 2"
            )

            sim_btn = gr.Button("Compute Similarity")
            sim_output = gr.Textbox(label="Similarity Score")

            sim_btn.click(
                compute_similarity,
                inputs=[text1, text2],
                outputs=sim_output
            )

        # ---------------- TAB 4 ----------------
        with gr.Tab("Zero-Shot Topic Classification"):

            zs_text = gr.Textbox(
                value="The government announced new policies on data privacy and cybersecurity.",
                label="Enter Text",
                lines=3
            )

            zs_labels = gr.Textbox(
                value="Politics, Technology, Sports, Health",
                label="Comma-Separated Categories"
            )

            zs_btn = gr.Button("Classify")
            zs_output = gr.Textbox(label="Category Confidence Scores")

            zs_btn.click(
                zero_shot_classify,
                inputs=[zs_text, zs_labels],
                outputs=zs_output
            )

        # ---------------- TAB 5 ----------------
        with gr.Tab("Summarization (Innovation)"):

            sum_input = gr.Textbox(
                value="Natural Language Processing enables computers to understand, interpret and generate human language. It powers applications such as chatbots, sentiment analysis, translation systems and much more.",
                label="Enter Long Text",
                lines=5
            )

            sum_btn = gr.Button("Summarize")
            sum_output = gr.Textbox(label="Summary")

            sum_btn.click(
                summarize_text,
                inputs=sum_input,
                outputs=sum_output
            )

demo.launch()