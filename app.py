import re
from pathlib import Path

import streamlit as st
import torch
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification

BASE_DIR = Path(__file__).resolve().parent

MIN_CHARS = 10
MAX_CHARS = 3000
MIN_ALPHA_RATIO = 0.5
MAX_LEN = 256

BERT_METRICS = {
    "Accuracy": 0.9076,
    "Precision": 0.8950,
    "Recall": 0.9235,
    "F1 Score": 0.9090,
}

def preprocess_bert(text):
    text = text.lower()
    text = re.sub(r"<.*?>", " ", text)
    text = re.sub(r"http\S+|www\S+", " ", text)

    contractions = {
        "don't": "do not", "doesn't": "does not", "didn't": "did not",
        "can't": "cannot", "won't": "will not", "isn't": "is not",
        "aren't": "are not", "wasn't": "was not", "weren't": "were not",
        "haven't": "have not", "hasn't": "has not", "hadn't": "had not",
        "wouldn't": "would not", "shouldn't": "should not", "couldn't": "could not",
        "it's": "it is", "that's": "that is", "there's": "there is",
        "i'm": "i am", "you're": "you are", "we're": "we are", "they're": "they are",
        "i've": "i have", "you've": "you have", "we've": "we have",
        "i'll": "i will", "you'll": "you will", "we'll": "we will",
    }

    for contraction, expanded in contractions.items():
        text = text.replace(contraction, expanded)

    return re.sub(r"\s+", " ", text).strip()

@st.cache_resource
def load_distilbert():
    model_path = BASE_DIR / "distilbert_model"

    try:
        tokenizer = DistilBertTokenizerFast.from_pretrained(str(model_path))
        model = DistilBertForSequenceClassification.from_pretrained(str(model_path))
        model.eval()
        return tokenizer, model

    except OSError as error:
        st.error(f"DistilBERT model not found at {model_path}: {error}")
        st.stop()

    except Exception as error:
        st.error(f"Failed to load DistilBERT model: {error}")
        st.stop()

def validate_review(text):
    stripped = text.strip()

    if not stripped:
        return False, "Please enter a review before analyzing."

    if len(stripped) < MIN_CHARS:
        return False, f"Review is too short. Please enter at least {MIN_CHARS} characters."

    if len(stripped) > MAX_CHARS:
        return False, f"Review is too long. Please keep it under {MAX_CHARS} characters."

    letters = sum(character.isalpha() for character in stripped)
    non_space = sum(not character.isspace() for character in stripped)

    if non_space == 0 or (letters / non_space) < MIN_ALPHA_RATIO:
        return False, "This doesn't look like a text review. Please enter a real sentence or two."

    words = re.findall(r"[a-zA-Z]+", stripped)
    real_words = [word for word in words if len(word) > 1]

    if len(real_words) < 3:
        return False, "Please write a slightly more detailed review (a few real words)."

    return True, None

st.set_page_config(
    page_title="Movie Review Sentiment Analysis",
    page_icon="🎬",
    layout="wide",
)

st.markdown(
    """
    <style>
    .result-card {
        border-radius: 12px;
        padding: 1.2rem 1.4rem;
        margin-top: 0.5rem;
        border: 1px solid rgba(128,128,128,0.25);
    }
    .result-positive {
        background: rgba(46, 204, 113, 0.12);
        border-color: rgba(46, 204, 113, 0.4);
    }
    .result-negative {
        background: rgba(231, 76, 60, 0.12);
        border-color: rgba(231, 76, 60, 0.4);
    }
    .result-label {
        font-size: 1.5rem;
        font-weight: 700;
    }
    .result-sub {
        color: rgba(128,128,128,0.9);
        font-size: 0.9rem;
        margin-top: 0.3rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

if "history" not in st.session_state:
    st.session_state.history = []

bert_tokenizer, bert_model = load_distilbert()

st.title("🎬 Movie Review Sentiment Analysis")

st.caption(
    "Enter a movie review below to determine whether its overall sentiment "
    "is Positive or Negative using the trained DistilBERT model."
)

with st.expander("🤖 About the Model", expanded=False):
    st.markdown("### DistilBERT")
    st.write("**Task:** Binary movie-review sentiment classification")
    st.write("**Process:** Text preprocessing → Tokenization → Fine-tuned DistilBERT")
    st.write("**Maximum sequence length:** 256 tokens")
    st.write("**Accuracy:** 90.76%")
    st.write(
        "DistilBERT was selected for the prototype because it achieved the "
        "strongest classification performance in the model evaluation."
    )

    st.info(
        "ℹ️ This prototype performs binary sentiment classification only: "
        "Positive or Negative. Neutral or mixed comments may be less accurately "
        "represented because Neutral is not a separate class."
    )

with st.expander("📊 Model Performance", expanded=False):
    performance_columns = st.columns(4)

    for column, (metric_name, value) in zip(performance_columns, BERT_METRICS.items()):
        with column:
            st.metric(metric_name, f"{value * 100:.2f}%")

with st.sidebar:
    st.header("ℹ️ About")
    st.write(
        "This application uses a fine-tuned **DistilBERT** model to classify "
        "movie reviews as **Positive** or **Negative**."
    )
    st.write(f"Accepted review length: **{MIN_CHARS}–{MAX_CHARS} characters**.")
    st.caption(
        "⚠️ Binary classification only: neutral or mixed comments may be less "
        "accurately represented because the model predicts only Positive or Negative."
    )

    st.divider()
    st.subheader("Try an example")

    example_reviews = {
        "😍 Positive review":
            "This film was an absolute masterpiece. The acting, the score, "
            "and the cinematography were excellent. I loved every moment.",
        "🤢 Negative review":
            "What a waste of time. The plot made no sense, the dialogue was "
            "terrible, and the movie was extremely boring."
    }

    for label, text in example_reviews.items():
        if st.button(label, use_container_width=True):
            st.session_state.example_text = text

    if st.session_state.history:
        st.divider()
        st.subheader("📜 Session history")
        st.caption(f"{len(st.session_state.history)} review(s) analyzed")

        if st.button("Clear history", use_container_width=True):
            st.session_state.history = []
            st.rerun()

review_text = st.text_area(
    "Your movie review:",
    value=st.session_state.get("example_text", ""),
    height=150,
    placeholder="Type or paste a movie review here...",
)

st.caption(f"{len(review_text)} / {MAX_CHARS} characters")

def render_result_card(label, confidence):
    css_class = "result-positive" if label == "Positive" else "result-negative"
    icon = "😊" if label == "Positive" else "☹️"

    html = (
        f'<div class="result-card {css_class}">'
        f'<div class="result-label">{icon} {label}</div>'
        f'<div class="result-sub">DistilBERT · Confidence {confidence * 100:.1f}%</div>'
        f'</div>'
    )

    st.markdown(html, unsafe_allow_html=True)
    st.progress(min(max(confidence, 0.0), 1.0))
    st.caption(
        "Confidence represents the model's prediction probability and does not "
        "guarantee that the prediction is correct."
    )

if st.button("Analyze Sentiment", type="primary"):
    is_valid, error_message = validate_review(review_text)

    if not is_valid:
        st.warning(error_message)

    else:
        try:
            with st.spinner("Analyzing review with DistilBERT..."):
                bert_text = preprocess_bert(review_text)

                inputs = bert_tokenizer(
                    bert_text,
                    truncation=True,
                    padding=True,
                    max_length=MAX_LEN,
                    return_tensors="pt",
                )

                with torch.no_grad():
                    outputs = bert_model(**inputs)
                    probabilities = torch.softmax(outputs.logits, dim=1)[0]
                    prediction = torch.argmax(probabilities).item()

                confidence = torch.max(probabilities).item()
                label = "Positive" if prediction == 1 else "Negative"

            st.subheader("Sentiment Result")
            render_result_card(label, confidence)

            if confidence < 0.60:
                st.warning(
                    "The model has relatively low confidence in this prediction. "
                    "The review may contain mixed, unclear, or ambiguous sentiment."
                )

            st.session_state.history.append({
                "Review": review_text.strip()[:70] + (
                    "..." if len(review_text.strip()) > 70 else ""
                ),
                "Prediction": label,
                "Confidence": f"{confidence * 100:.1f}%",
            })

        except Exception as error:
            st.error(f"Sentiment prediction failed: {error}")

if st.session_state.history:
    st.divider()
    st.subheader("📜 Analysis History (This Session)")

    st.dataframe(
        st.session_state.history,
        use_container_width=True,
        hide_index=True,
    )
