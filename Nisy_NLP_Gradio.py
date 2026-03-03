#!/usr/bin/env python3
"""
Gradio NLP Toolkit - multi-tab application

Tabs implemented:
- Text Preprocessing (Tokenization): compare NLTK word-tokenize vs Hugging Face BPE (gpt2)
- Sentiment & Emotion Analysis: transformer sentiment pipeline
- Semantic Similarity: sentence-transformers cosine similarity
- Zero-Shot Topic Classification: NLI-based zero-shot classifier
- Innovation: Summarization (distilbart)

This file lazily loads models to keep startup responsive.
"""

import io
import math
import os
import re
import socket
from typing import List, Tuple

import gradio as gr
import numpy as np
import pandas as pd
from PIL import Image, ImageOps

# Optional heavy deps; import when needed
try:
	import nltk
except Exception:
	nltk = None

try:
	from transformers import pipeline, AutoTokenizer
except Exception:
	pipeline = None
	AutoTokenizer = None
try:
	from transformers import AutoModelForSeq2SeqLM
except Exception:
	AutoModelForSeq2SeqLM = None

try:
	from langdetect import detect_langs, detect
except Exception:
	detect_langs = None
	detect = None
try:
	from dateparser.search import search_dates
except Exception:
	search_dates = None

# Friendly label mappings
LABEL_MAP = {
	"PER": "Person",
	"PERSON": "Person",
	"LOC": "Location",
	"LOCATION": "Location",
	"ORG": "Organization",
	"ORGANIZATION": "Organization",
	"MISC": "Misc",
	"DATE": "Date",
	"TIME": "Time",
	"GPE": "Location",
}

# Common language code -> name map
LANG_NAME = {
	"en": "English",
	"en-us": "English (US)",
	"en-gb": "English (UK)",
	"fr": "French",
	"es": "Spanish",
	"de": "German",
	"it": "Italian",
	"pt": "Portuguese",
	"pt-br": "Portuguese (Brazil)",
	"ru": "Russian",
	"zh-cn": "Chinese (Simplified)",
	"zh-tw": "Chinese (Traditional)",
	"zh": "Chinese",
	"ja": "Japanese",
	"ko": "Korean",
	"ar": "Arabic",
	"hi": "Hindi",
	"bn": "Bengali",
	"pa": "Punjabi",
	"vi": "Vietnamese",
	"id": "Indonesian",
	"nl": "Dutch",
	"sv": "Swedish",
	"no": "Norwegian",
	"da": "Danish",
	"fi": "Finnish",
	"pl": "Polish",
	"tr": "Turkish",
	"cs": "Czech",
	"el": "Greek",
	"he": "Hebrew",
	"uk": "Ukrainian",
	"bg": "Bulgarian",
	"ro": "Romanian",
	"hu": "Hungarian",
	"sk": "Slovak",
	"sl": "Slovenian",
	"hr": "Croatian",
	"sr": "Serbian",
	"lt": "Lithuanian",
	"lv": "Latvian",
	"et": "Estonian",
	"af": "Afrikaans",
	"sw": "Swahili",
	"tl": "Tagalog",
	"ms": "Malay",
}

try:
	from sentence_transformers import SentenceTransformer
except Exception:
	SentenceTransformer = None


def ensure_nltk_punkt():
	if nltk is None:
		return False
	try:
		nltk.data.find("tokenizers/punkt")
	except LookupError:
		nltk.download("punkt", quiet=True)
	return True


# -----------------
# Tokenization tab
# -----------------

HF_TK = None

def get_hf_tokenizer(model_name: str = "gpt2"):
	global HF_TK
	if AutoTokenizer is None:
		return None
	if HF_TK is None:
		HF_TK = AutoTokenizer.from_pretrained(model_name)
	return HF_TK


def tokenize_nltk(text: str) -> List[str]:
	if not text:
		return []
	if nltk is None:
		return ["NLTK not installed"]
	try:
		ensure_nltk_punkt()
		return nltk.word_tokenize(text)
	except LookupError:
		# try to download and retry
		try:
			nltk.download("punkt", quiet=True)
			return nltk.word_tokenize(text)
		except Exception:
			# final fallback: simple regex tokenization
			return re.findall(r"\w+|[^\s\w]", text)
	except Exception:
		return re.findall(r"\w+|[^\s\w]", text)


def tokenize_hf(text: str, model_name: str = "gpt2") -> List[str]:
	if not text:
		return []
	tk = get_hf_tokenizer(model_name)
	if tk is None:
		return ["transformers not installed"]
	return tk.tokenize(text)


def compare_tokenizers(text: str) -> Tuple[str, str]:
	nltk_tokens = tokenize_nltk(text)
	hf_tokens = tokenize_hf(text)
	# If NLTK returned an install message, bubble that up
	if isinstance(nltk_tokens, list) and len(nltk_tokens) == 1 and nltk_tokens[0].startswith("NLTK not installed"):
		nltk_md = "**NLTK (word-level)**\n\nNLTK not installed in environment. Install with `pip install nltk` and rerun."
	else:
		nltk_md = f"**NLTK (word-level)**\n\nTotal tokens: {len(nltk_tokens)}\n\n{nltk_tokens}"
	hf_md = f"**Hugging Face (gpt2 BPE)**\n\nTotal tokens: {len(hf_tokens)}\n\n{hf_tokens}"
	return nltk_md, hf_md


# -----------------
# Sentiment tab
# -----------------

SENT_PIPE = None

def get_sentiment_pipeline(model_name: str = "distilbert-base-uncased-finetuned-sst-2-english"):
	global SENT_PIPE
	if pipeline is None:
		return None
	if SENT_PIPE is None:
		# prefer CPU device (-1) for compatibility
		try:
			SENT_PIPE = pipeline("sentiment-analysis", model=model_name, device=-1)
		except TypeError:
			# some pipeline versions don't accept device; fall back
			SENT_PIPE = pipeline("sentiment-analysis", model=model_name)
	return SENT_PIPE


def sentiment_analyze(text: str) -> str:
	if not text:
		return "No text provided."
	p = get_sentiment_pipeline()
	if p is None:
		return "transformers not installed"
	results = p(text, return_all_scores=True)
	# Normalize to a list of score dicts
	scores = []
	if isinstance(results, list) and results:
		first = results[0]
		if isinstance(first, list):
			scores = first
		elif isinstance(first, dict):
			scores = results
		else:
			# unexpected structure, convert items to strings
			scores = [results]
	elif isinstance(results, dict):
		scores = [results]
	else:
		scores = []
	# Build label/score pairs
	pairs = []
	for r in scores:
		if isinstance(r, dict):
			label = r.get("label", "")
			score = r.get("score")
			try:
				score = float(score)
			except Exception:
				score = None
			pairs.append((label, score))

	if not pairs:
		return "No scores returned."

	# If numeric scores don't sum to ~1, apply softmax to convert logits to probabilities
	numeric = [s if s is not None else 0.0 for _, s in pairs]
	total = sum(numeric)
	def softmax(xs):
		ex = [math.exp(x) for x in xs]
		s = sum(ex)
		return [e / s for e in ex]

	# Special-case: if the pipeline returned only one label (common),
	# infer the complementary binary score so both Positive and Negative are shown.
	if len(pairs) == 1:
		s = numeric[0]
		# If s looks like a probability in [0,1], use it directly; else apply sigmoid
		if 0.0 <= s <= 1.0:
			prob_label = s
		else:
			# treat as logit
			prob_label = 1.0 / (1.0 + math.exp(-s)) if isinstance(s, (int, float)) else 0.0
		# Determine primary label name and complementary label
		label_raw = pairs[0][0].upper() if pairs[0][0] else ""
		if "NEG" in label_raw:
			primary, secondary = "NEGATIVE", "POSITIVE"
		else:
			primary, secondary = "POSITIVE", "NEGATIVE"
		probs = [prob_label, 1 - prob_label]
		labels_out = [primary, secondary]
	else:
		if not math.isclose(total, 1.0, rel_tol=1e-2):
			probs = softmax(numeric)
		else:
			probs = [x if x is not None else 0.0 for x in numeric]
		labels_out = [lab for lab, _ in pairs]

	md_lines = []
	# If we constructed labels_out for single-label case, use it; otherwise use pairs' labels
	if 'labels_out' in locals():
		for lab, prob in zip(labels_out, probs):
			md_lines.append(f"- **{lab}**: {prob*100:.1f}%")
	else:
		for (label, _), prob in zip(pairs, probs):
			md_lines.append(f"- **{label}**: {prob*100:.1f}%")
	return "\n".join(md_lines)


# -----------------
# Semantic similarity tab
# -----------------

SENTENCE_MODEL = None

def get_sentence_model(model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
	global SENTENCE_MODEL
	if SentenceTransformer is None:
		return None
	if SENTENCE_MODEL is None:
		SENTENCE_MODEL = SentenceTransformer(model_name)
	return SENTENCE_MODEL


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
	if a is None or b is None:
		return 0.0
	denom = (np.linalg.norm(a) * np.linalg.norm(b))
	if denom == 0:
		return 0.0
	return float(np.dot(a, b) / denom)


def semantic_similarity(a: str, b: str) -> str:
	if not a or not b:
		return "Both inputs required."
	model = get_sentence_model()
	if model is None:
		return "sentence-transformers not installed"
	emb = model.encode([a, b])
	score = cosine_sim(emb[0], emb[1])
	return f"Cosine similarity: {score:.4f}"


# -----------------
# Zero-shot classification tab
# -----------------

ZS_PIPE = None

def get_zero_shot_pipeline(model_name: str = "facebook/bart-large-mnli"):
	global ZS_PIPE
	if pipeline is None:
		return None
	if ZS_PIPE is None:
		ZS_PIPE = pipeline("zero-shot-classification", model=model_name)
	return ZS_PIPE


def zero_shot_classify(text: str, labels_csv: str) -> str:
	if not text or not labels_csv:
		return "Text and comma-separated labels required."
	labels = [l.strip() for l in labels_csv.split(",") if l.strip()]
	if not labels:
		return "No valid labels provided."
	pipe = get_zero_shot_pipeline()
	if pipe is None:
		return "transformers not installed"
	out = pipe(text, candidate_labels=labels)
	lines = [f"- **{lab}**: {score*100:.1f}%" for lab, score in zip(out["labels"], out["scores"]) ]
	return "\n".join(lines)


# -----------------
# NER tab
# -----------------

NER_PIPE = None

def get_ner_pipeline(model_name: str = "dbmdz/bert-large-cased-finetuned-conll03-english"):
	global NER_PIPE
	if pipeline is None:
		return None
	if NER_PIPE is None:
		try:
			# aggregation_strategy may vary by transformers version
			NER_PIPE = pipeline("ner", model=model_name, aggregation_strategy="simple")
		except Exception:
			try:
				NER_PIPE = pipeline("ner", model=model_name, grouped_entities=True)
			except Exception:
				NER_PIPE = pipeline("ner", model=model_name)
	return NER_PIPE


def ner_extract(text: str) -> str:
	if not text:
		return "No text provided."
	p = get_ner_pipeline()
	if p is None:
		return "transformers not installed"
	try:
		ents = p(text)
		# ents may be list of dicts with entity/group
		lines = []
		for e in ents:
			if isinstance(e, dict):
				label = e.get("entity_group") or e.get("entity") or e.get("label")
				# normalize label: remove common prefixes and uppercase
				if isinstance(label, str):
					# remove leading B- or I- if present, then uppercase
					label_norm = label.upper().lstrip("BI-")
				else:
					label_norm = str(label)
				word = e.get("word") or e.get("text") or e.get("entity")
				score = e.get("score")
				# map short labels to friendly names
				friendly = LABEL_MAP.get(label_norm, None)
				display_label = friendly if friendly is not None else (label_norm.title() if isinstance(label_norm, str) else str(label_norm))
				if score is not None:
					lines.append(f"- **{display_label}**: {word} ({score*100:.1f}%)")
				else:
					lines.append(f"- **{display_label}**: {word}")

		# Additionally try to extract date-like expressions via regex (fallback)
		date_patterns = [
			r"\b\d{4}[/-]\d{1,2}[/-]\d{1,2}\b",  # 2020-01-31
			r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b",  # 01/31/2020 or 1-1-20
			r"\b(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\b\s+\d{1,2}(?:st|nd|rd|th)?,?\s+\d{4}",
			r"\b\d{1,2}\s+(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?),?\s+\d{4}\b",
		]
		date_matches = []
		for pat in date_patterns:
			found = re.findall(pat, text, flags=re.IGNORECASE)
			for f in found:
				if f and f not in date_matches:
					date_matches.append(f)

		# Merge regex-found dates into the entity lines if not already present
		for d in date_matches:
			# avoid duplicates if an entity already contains the date string
			if not any(d in l for l in lines):
				lines.append(f"- **DATE**: {d}")

		# If regex didn't find dates or to find additional formats, try dateparser.search.search_dates
		if search_dates is not None:
			try:
				found = search_dates(text, settings={"STRICT_PARSING": True})
				if found:
					for orig, dt in found:
						if orig and not any(orig in l for l in lines):
							lines.append(f"- **DATE**: {orig}")
			except Exception:
				# ignore dateparser failures
				pass

		return "\n".join(lines) if lines else "No entities found."
	except Exception as e:
		return f"NER error: {e}"


# -----------------
# Language detection tab
# -----------------

def detect_language(text: str) -> str:
	if not text:
		return "No text provided."
	# Try langdetect.detect_langs if available
	if detect_langs is not None:
		try:
			probs = detect_langs(text)
			if probs:
				lines = []
				for p in probs:
					code = p.lang.lower()
					name = LANG_NAME.get(code, p.lang)
					lines.append(f"- {name}: {p.prob*100:.1f}%")
				return "\n".join(lines)
		except Exception:
			pass

	# Fallback: detect() which returns a single language
	if detect is not None:
		try:
			lang = detect(text)
			name = LANG_NAME.get(lang.lower(), lang)
			return f"- {name}: 100%"
		except Exception:
			pass

	# Final heuristic fallback: simple stopword matching for a few common languages
	text_l = text.lower()
	stopwords = {
		"en": {"the", "and", "is", "in", "to", "of"},
		"es": {"el", "y", "es", "en", "la", "de"},
		"fr": {"le", "et", "est", "en", "la", "de"},
	}
	scores = {}
	words = re.findall(r"\w+", text_l)
	for code, stops in stopwords.items():
		scores[code] = sum(1 for w in words if w in stops)
	best = max(scores.items(), key=lambda x: x[1])
	if best[1] > 0:
		# approximate confidence by proportion of matched stopwords
		conf = best[1] / max(1, len(words))
		lang_name = LANG_NAME.get(best[0].lower(), best[0])
		return f"- {lang_name}: {conf*100:.1f}% (heuristic)"

	return "Could not reliably detect language. Install `langdetect` for better results."


# -----------------
# Question Answering tab
# -----------------

QA_PIPE = None

def get_qa_pipeline(model_name: str = "distilbert-base-cased-distilled-squad"):
	global QA_PIPE
	if pipeline is None:
		return None
	if QA_PIPE is None:
		QA_PIPE = pipeline("question-answering", model=model_name)
	return QA_PIPE


def answer_question(question: str, context: str) -> str:
	if not question or not context:
		return "Both question and context are required."
	p = get_qa_pipeline()
	if p is None:
		return "transformers not installed"
	try:
		out = p(question=question, context=context)
		if isinstance(out, dict):
			ans = out.get("answer")
			score = out.get("score")
			return f"Answer: {ans}\nConfidence: {score*100:.1f}%"
		return str(out)
	except Exception as e:
		return f"QA error: {e}"


# -----------------
# Innovation: Summarization
# -----------------

SUM_PIPE = None
SUM_PIPE_ERROR = None

def get_summarizer(model_name: str = "sshleifer/distilbart-cnn-12-6"):
	global SUM_PIPE
	global SUM_PIPE_ERROR
	if pipeline is None:
		return None
	if SUM_PIPE is None:
		# Prefer tasks that map to seq2seq models; avoid 'text-generation' which may load causal LM classes
		tasks_to_try = ["summarization", "text2text-generation"]
		last_exc = None
		hf_token = os.environ.get("HF_TOKEN")
		for task in tasks_to_try:
			try:
				# prefer CPU device (-1)
				try:
					if hf_token:
						SUM_PIPE = pipeline(task, model=model_name, device=-1, use_auth_token=hf_token)
					else:
						SUM_PIPE = pipeline(task, model=model_name, device=-1)
				except TypeError:
					# older/newer pipeline API may not accept device or use_auth_token
					if hf_token:
						SUM_PIPE = pipeline(task, model=model_name, use_auth_token=hf_token)
					else:
						SUM_PIPE = pipeline(task, model=model_name)
				# success
				SUM_PIPE_ERROR = None
				break
			except Exception as e:
				last_exc = e
				SUM_PIPE = None
				continue
		if SUM_PIPE is None:
			# try direct seq2seq model+tokenizer as a fallback
			last_model_exc = None
			if AutoTokenizer is not None and AutoModelForSeq2SeqLM is not None:
				try:
					tok = AutoTokenizer.from_pretrained(model_name)
					mdl = AutoModelForSeq2SeqLM.from_pretrained(model_name)
					# put model in eval mode
					try:
						mdl.eval()
					except Exception:
						pass
					SUM_PIPE = ("model", tok, mdl)
					SUM_PIPE_ERROR = None
				except Exception as e:
					last_model_exc = e
			if SUM_PIPE is None:
				SUM_PIPE_ERROR = f"Could not create summarization pipeline: {last_exc}; model fallback error: {last_model_exc}"
	return SUM_PIPE


def summarize_text(text: str, max_length: int = 60) -> str:
	if not text:
		return "No text provided."
	pipe = get_summarizer()
	if pipe is None:
		# Provide the pipeline error if available
		if 'SUM_PIPE_ERROR' in globals() and SUM_PIPE_ERROR:
			return f"Summarization pipeline error: {SUM_PIPE_ERROR}\nYou can set HF_TOKEN env var for authenticated downloads to avoid rate limits."
		return "transformers not installed or pipeline task unsupported"
	# If we returned a tuple with model, use it directly for seq2seq generation
	try:
		if isinstance(pipe, tuple) and pipe[0] == "model":
			_, tok, mdl = pipe
			try:
				import torch
				device = torch.device("cpu")
				mdl.to(device)
				inputs = tok(text, return_tensors="pt", truncation=True)
				inputs = {k: v.to(device) for k, v in inputs.items()}
				# use beam search for better summaries on CPU
				outputs = mdl.generate(**inputs, max_length=max_length, min_length=10, num_beams=4, early_stopping=True)
				summary = tok.decode(outputs[0], skip_special_tokens=True, clean_up_tokenization_spaces=True)
				return summary
			except Exception as me:
				return f"Summarization model error: {me}"
		else:
			out = pipe(text, max_length=max_length, min_length=10)
			# common output shapes: [{'summary_text': ...}] or [{'generated_text': ...}]
			if isinstance(out, list) and out:
				first = out[0]
				if isinstance(first, dict):
					if "summary_text" in first:
						return first["summary_text"]
					if "generated_text" in first:
						return first["generated_text"]
				return str(first)
			return str(out)
	except Exception as e:
		return f"Summarization error: {e}"


# -----------------
# Utility tabs (image, data, chat) - kept minimal
# -----------------

def process_image(img: Image.Image) -> Image.Image:
	if img is None:
		return None
	return ImageOps.grayscale(img)


def preview_csv(file_obj):
	if file_obj is None:
		return pd.DataFrame()
	try:
		df = pd.read_csv(file_obj.name)
	except Exception:
		try:
			file_obj.seek(0)
			df = pd.read_csv(io.BytesIO(file_obj.read()))
		except Exception as e:
			return pd.DataFrame({"error": [f"Error reading CSV: {e}"]})
	return df.head(10)


def build_ui():
	example = "I am Lady Nisy and I love to explore and also love to learn and try out new things."
	with gr.Blocks(title="Gradio NLP Toolkit") as demo:
		gr.Markdown("# Nisy Natural Language Processing Gradio Toolkit\n\nA multi-tab application showcasing various NLP tasks using Hugging Face transformers and other libraries. Each tab demonstrates a different NLP capability, with lazy loading of models for responsiveness.")
		with gr.Tabs():
			with gr.TabItem("Text Preprocessing (Tokenization)"):
				with gr.Row():
					txt = gr.Textbox(lines=5, value=example, label="Input text")
					tk_btn = gr.Button("Tokenize")
				with gr.Row():
					nltk_out = gr.Markdown(label="NLTK tokens")
					hf_out = gr.Markdown(label="Hugging Face tokens")
				tk_btn.click(compare_tokenizers, inputs=txt, outputs=[nltk_out, hf_out])

			with gr.TabItem("Sentiment & Emotion Analysis"):
				s_txt = gr.Textbox(lines=4, value="I almost like it", label="Input text")
				s_btn = gr.Button("Analyze Sentiment")
				s_out = gr.Markdown()
				s_btn.click(sentiment_analyze, inputs=s_txt, outputs=s_out)

			with gr.TabItem("Semantic Similarity"):
				a = gr.Textbox(lines=2, value="I like pizza.", label="Text A")
				b = gr.Textbox(lines=2, value="I enjoy eating Italian food.", label="Text B")
				sim_btn = gr.Button("Compute Similarity")
				sim_out = gr.Markdown()
				sim_btn.click(semantic_similarity, inputs=[a, b], outputs=sim_out)

			with gr.TabItem("Zero-Shot Topic Classification"):
				z_txt = gr.Textbox(lines=3, value="The government passed a new regulation on data privacy.", label="Text")
				with gr.Row():
					z_labels = gr.Textbox(lines=1, value="Politics, Tech, Sports, Business, Health", label="Comma-separated labels", placeholder="Enter labels separated by commas")
					z_btn = gr.Button("Classify")
				z_out = gr.Markdown()
				z_btn.click(zero_shot_classify, inputs=[z_txt, z_labels], outputs=z_out)

			with gr.TabItem("Summarization"):
				sum_txt = gr.Textbox(lines=6, value="Long text to summarize.\n\nThe global shift toward remote work environments has fundamentally restructured the traditional corporate landscape, moving away from rigid office-centric models to more fluid, asynchronous workflows. While critics initially feared a significant dip in collaborative output and employee accountability, empirical data from the last few years suggests that flexibility often leads to higher job satisfaction and decreased overhead costs for massive enterprises. However, this transition is not without its hurdles, as digital fatigue and the erosion of spontaneous 'watercooler' innovation remain pressing challenges for HR departments worldwide. Ultimately, the success of the hybrid era depends not on the physical location of the staff, but on the robustness of the digital infrastructure and the intentionality of the leadership in fostering a cohesive company culture.")
				sum_btn = gr.Button("Summarize")
				sum_out = gr.Textbox(label="Summary")
				sum_btn.click(summarize_text, inputs=sum_txt, outputs=sum_out)

			with gr.TabItem("Named Entity Recognition (NER)"):
				ner_txt = gr.Textbox(lines=4, value="I am Nisy, I live in Lille and I completed this project on 2nd March, 2026", label="Text for NER")
				ner_btn = gr.Button("Extract Entities")
				ner_out = gr.Markdown()
				ner_btn.click(ner_extract, inputs=ner_txt, outputs=ner_out)

			with gr.TabItem("Language Detection"):
				lang_txt = gr.Textbox(lines=4, value="Bonjour, je suis mon amie, et je suis en train de terminer ce projet le 2 mars 2026.", label="Text for language detection")
				lang_btn = gr.Button("Detect Language")
				lang_out = gr.Markdown()
				lang_btn.click(detect_language, inputs=lang_txt, outputs=lang_out)

			with gr.TabItem("Question Answering"):
				qa_context = gr.Textbox(lines=6, value="Nisy Adjei, a master's student at Junia is going to the mall." + "\n\n.", label="Context")
				qa_question = gr.Textbox(lines=2, value="Who is Nisy?", label="Question")
				qa_btn = gr.Button("Answer")
				qa_out = gr.Textbox(label="Answer")
				qa_btn.click(answer_question, inputs=[qa_question, qa_context], outputs=qa_out)

		return demo


def choose_server_port(preferred: int = 7860) -> int:
	"""Choose a port preferring `preferred` or the `GRADIO_SERVER_PORT` env var; fall back to an OS-assigned free port."""
	env = os.environ.get("GRADIO_SERVER_PORT")
	try:
		preferred = int(env) if env is not None else preferred
	except Exception:
		pass

	# Try preferred port first
	s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
	try:
		s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
		s.bind(("", preferred))
		s.close()
		return preferred
	except OSError:
		s.close()
	# Ask OS for a free port
	s2 = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
	s2.bind(("", 0))
	port = s2.getsockname()[1]
	s2.close()
	return port


if __name__ == "__main__":
	demo = build_ui()
	port = choose_server_port(7860)
	print(f"Starting Gradio on port {port}")
	demo.launch(server_name="0.0.0.0", server_port=port, share=False)
