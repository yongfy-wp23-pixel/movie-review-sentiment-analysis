import re
from pathlib import Path

import streamlit as st
import torch
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
from langdetect import detect, LangDetectException
from deep_translator import GoogleTranslator

BASE_DIR = Path(__file__).resolve().parent

MIN_CHARS = 10
MAX_CHARS = 1500
MIN_ALPHA_RATIO = 0.5
MAX_LEN = 256

METRICS = {
    "Accuracy": 90.76,
    "Precision": 89.50,
    "Recall": 92.35,
    "F1 Score": 90.90,
}

def preprocess(text):
    text = text.lower()
    text = re.sub(r"<.*?>", " ", text)
    text = re.sub(r"http\S+|www\S+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def detect_language(text):
    if re.search(r"[\u3040-\u30ff]", text):
        return "ja"
    if re.search(r"[\uac00-\ud7af]", text):
        return "ko"
    if re.search(r"[\u4e00-\u9fff]", text):
        return "zh-cn"
    try:
        return detect(text)
    except LangDetectException:
        return "unknown"

def language_name(code):
    names = {
        "en":"English","zh-cn":"Chinese","zh-tw":"Chinese","ja":"Japanese",
        "ko":"Korean","ms":"Malay","id":"Indonesian","th":"Thai","vi":"Vietnamese",
        "fr":"French","de":"German","es":"Spanish","it":"Italian","pt":"Portuguese",
        "ru":"Russian","ar":"Arabic","hi":"Hindi","tl":"Filipino","unknown":"Unknown"
    }
    return names.get(code, code.upper())

def detect_and_translate(text):
    lang = detect_language(text)
    if lang == "en":
        return text, lang, False
    if lang == "unknown":
        raise ValueError("Unable to detect the language.")
    translated = GoogleTranslator(source="auto", target="en").translate(text)
    return translated, lang, True

@st.cache_resource
def load_model():
    model_dir = BASE_DIR / "distilbert_model"
    tokenizer = DistilBertTokenizerFast.from_pretrained(str(model_dir))
    model = DistilBertForSequenceClassification.from_pretrained(str(model_dir))
    model.eval()
    return tokenizer, model

def validate(text):
    text = text.strip()
    if not text:
        return False, "Please enter a movie review."
    if len(text) < MIN_CHARS:
        return False, f"Please enter at least {MIN_CHARS} characters."
    if len(text) > MAX_CHARS:
        return False, f"Please keep the review under {MAX_CHARS} characters."
    letters = sum(c.isalpha() for c in text)
    non_space = sum(not c.isspace() for c in text)
    if non_space == 0 or letters / non_space < MIN_ALPHA_RATIO:
        return False, "Please enter a meaningful movie review."
    return True, ""

st.set_page_config(page_title="Movie Review Sentiment Analysis", page_icon="🎬", layout="wide")

st.markdown("""
<style>
.card{border-radius:12px;padding:18px;border:1px solid rgba(128,128,128,.25);}
.pos{background:rgba(46,204,113,.12);border-color:rgba(46,204,113,.4);}
.neg{background:rgba(231,76,60,.12);border-color:rgba(231,76,60,.4);}
</style>
""", unsafe_allow_html=True)

if "history" not in st.session_state: st.session_state.history=[]
if "example" not in st.session_state: st.session_state.example=""

tokenizer, model = load_model()

st.title("🎬 Movie Review Sentiment Analysis")
st.caption("Enter a movie review in English or another language. Non-English reviews are automatically translated to English before DistilBERT performs sentiment analysis.")

with st.expander("🤖 About the Model"):
    st.write("**Model:** Fine-tuned DistilBERT")
    st.write("**Task:** Binary sentiment classification (Positive / Negative)")
    st.write("**Maximum sequence length:** 256 tokens")
    st.info("Neutral or mixed reviews may be less accurately represented because the model predicts only Positive or Negative.")

with st.expander("📊 Model Performance"):
    cols=st.columns(4)
    for c,(k,v) in zip(cols,METRICS.items()):
        c.metric(k,f"{v:.2f}%")

with st.sidebar:
    st.header("ℹ️ About")
    st.write("This application uses a fine-tuned DistilBERT model to classify movie reviews as Positive or Negative.")
    st.write("Non-English reviews are automatically translated before classification.")
    st.write(f"Accepted review length: **{MIN_CHARS}-{MAX_CHARS} characters**.")
    st.caption("⚠️ Binary classification only.")
    st.divider()
    st.subheader("Try an example")
    examples={
        "😍 Positive review":"This film was amazing. The acting, music, and cinematography were excellent.",
        "🤢 Negative review":"This movie was boring, the plot made no sense, and I regretted watching it."
    }
    for label,text in examples.items():
        if st.button(label,use_container_width=True):
            st.session_state.example=text
            st.rerun()
    if st.session_state.history:
        st.divider()
        st.subheader("📜 Session history")
        st.caption(f"{len(st.session_state.history)} review(s) analyzed")
        if st.button("Clear history",use_container_width=True):
            st.session_state.history=[]
            st.rerun()

review=st.text_area("Your movie review:", value=st.session_state.example, height=160)
st.caption(f"{len(review)} / {MAX_CHARS} characters")
st.caption("🌐 Non-English reviews will be automatically translated to English before analysis.")

if st.button("Analyze Sentiment", type="primary"):
    ok,msg=validate(review)
    if not ok:
        st.warning(msg)
    else:
        with st.spinner("Analyzing sentiment..."):
            text,lang,translated=detect_and_translate(review)
            if translated:
                st.info(f"🌐 Detected language: **{language_name(lang)}**. The review was translated to English before sentiment analysis.")
                st.markdown("**Translated English review:**")
                st.write(text)
            else:
                st.caption("🌐 Detected language: **English**")
            text=preprocess(text)
            inputs=tokenizer(text,truncation=True,padding=True,max_length=MAX_LEN,return_tensors="pt")
            with torch.no_grad():
                probs=torch.softmax(model(**inputs).logits,dim=1)[0]
            pred=torch.argmax(probs).item()
            conf=float(torch.max(probs))
            label="Positive" if pred==1 else "Negative"
            css="pos" if label=="Positive" else "neg"
            icon="😊" if label=="Positive" else "☹️"
            st.subheader("Sentiment Result")
            st.markdown(f'<div class="card {css}"><h2>{icon} {label}</h2><p>DistilBERT · Confidence {conf*100:.1f}%</p></div>', unsafe_allow_html=True)
            st.progress(conf)
            if conf<0.60:
                st.warning("The model has relatively low confidence in this prediction.")
            st.session_state.history.append({
                "Review":review[:60]+("..." if len(review)>60 else ""),
                "Language":language_name(lang),
                "Translated":"Yes" if translated else "No",
                "Prediction":label,
                "Confidence":f"{conf*100:.1f}%"
            })

if st.session_state.history:
    st.divider()
    st.subheader("📜 Analysis History (This Session)")
    st.dataframe(st.session_state.history, use_container_width=True, hide_index=True)
