import re
from pathlib import Path

import streamlit as st
import torch
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
from langdetect import detect, LangDetectException
from deep_translator import GoogleTranslator


# =========================================================
# PROJECT PATH
# =========================================================

BASE_DIR = Path(__file__).resolve().parent


# =========================================================
# BUSINESS RULES
# =========================================================

MIN_CHARS = 5
MAX_CHARS = 1500
MIN_ALPHA_RATIO = 0.5
MAX_LEN = 256


# =========================================================
# DISTILBERT PERFORMANCE
# =========================================================

BERT_METRICS = {
    "Accuracy": 0.9076,
    "Precision": 0.8950,
    "Recall": 0.9235,
    "F1 Score": 0.9090,
}


# =========================================================
# DISTILBERT PREPROCESSING
# =========================================================

def preprocess_bert(text):
    text = text.lower()

    # Remove HTML tags
    text = re.sub(r"<.*?>", " ", text)

    # Remove URLs
    text = re.sub(r"http\S+|www\S+", " ", text)

    # Expand common English contractions
    contractions = {
        "don't": "do not",
        "doesn't": "does not",
        "didn't": "did not",
        "can't": "cannot",
        "won't": "will not",
        "isn't": "is not",
        "aren't": "are not",
        "wasn't": "was not",
        "weren't": "were not",
        "haven't": "have not",
        "hasn't": "has not",
        "hadn't": "had not",
        "wouldn't": "would not",
        "shouldn't": "should not",
        "couldn't": "could not",
        "it's": "it is",
        "that's": "that is",
        "there's": "there is",
        "i'm": "i am",
        "you're": "you are",
        "we're": "we are",
        "they're": "they are",
        "i've": "i have",
        "you've": "you have",
        "we've": "we have",
        "i'll": "i will",
        "you'll": "you will",
        "we'll": "we will",
    }

    for contraction, expanded in contractions.items():
        text = text.replace(contraction, expanded)

    # Remove extra spaces
    text = re.sub(r"\s+", " ", text).strip()

    return text


# =========================================================
# LANGUAGE DETECTION AND TRANSLATION
# =========================================================

def detect_and_translate(text):
    """
    Detect the input language.
    If the review is not English, translate it to English
    before sending it to DistilBERT.
    """

    try:
        detected_language = detect(text)

    except LangDetectException:
        detected_language = "unknown"

    # Already English
    if detected_language == "en":
        return text, detected_language, False

    # Language cannot be detected
    if detected_language == "unknown":
        raise ValueError(
            "The language could not be detected. "
            "Please enter a clearer movie review."
        )

    # Translate non-English input to English
    try:
        translated_text = GoogleTranslator(
            source="auto",
            target="en"
        ).translate(text)

        if not translated_text or not translated_text.strip():
            raise ValueError(
                "Translation returned an empty result."
            )

        return translated_text, detected_language, True

    except Exception as error:
        raise RuntimeError(
            "Translation failed. Please try again "
            "or enter the review in English."
        ) from error


# =========================================================
# LOAD DISTILBERT MODEL
# =========================================================

@st.cache_resource
def load_distilbert():

    model_path = BASE_DIR / "distilbert_model"

    try:
        tokenizer = (
            DistilBertTokenizerFast
            .from_pretrained(str(model_path))
        )

        model = (
            DistilBertForSequenceClassification
            .from_pretrained(str(model_path))
        )

        model.eval()

        return tokenizer, model

    except OSError as error:
        st.error(
            f"DistilBERT model not found at {model_path}: {error}"
        )
        st.stop()

    except Exception as error:
        st.error(
            f"Failed to load DistilBERT model: {error}"
        )
        st.stop()


# =========================================================
# INPUT VALIDATION
# =========================================================

def validate_review(text):

    stripped = text.strip()

    # Empty input
    if not stripped:
        return (
            False,
            "Please enter a review before analyzing."
        )

    # Minimum length
    if len(stripped) < MIN_CHARS:
        return (
            False,
            f"Review is too short. Please enter at least "
            f"{MIN_CHARS} characters."
        )

    # Maximum length
    if len(stripped) > MAX_CHARS:
        return (
            False,
            f"Review is too long. Please keep it under "
            f"{MAX_CHARS} characters."
        )

    # Multilingual validation:
    # Check that enough of the input consists of alphabetic characters.
    # This works better than requiring words separated by spaces,
    # because languages such as Chinese, Japanese and Thai may not
    # separate words in the same way as English.
    letters = sum(
        character.isalpha()
        for character in stripped
    )

    non_space = sum(
        not character.isspace()
        for character in stripped
    )

    if (
        non_space == 0
        or
        (letters / non_space) < MIN_ALPHA_RATIO
    ):
        return (
            False,
            "This doesn't look like a valid text review. "
            "Please enter a meaningful movie review."
        )

    return True, None


# =========================================================
# STREAMLIT PAGE SETTINGS
# =========================================================

st.set_page_config(
    page_title="Movie Review Sentiment Analysis",
    page_icon="🎬",
    layout="wide"
)


# =========================================================
# CUSTOM STYLE
# =========================================================

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
    unsafe_allow_html=True
)


# =========================================================
# SESSION STATE
# =========================================================

if "history" not in st.session_state:
    st.session_state.history = []

if "example_text" not in st.session_state:
    st.session_state.example_text = ""


# =========================================================
# LOAD MODEL
# =========================================================

bert_tokenizer, bert_model = load_distilbert()


# =========================================================
# TITLE
# =========================================================

st.title(
    "🎬 Movie Review Sentiment Analysis"
)

st.caption(
    "Enter a movie review in English or another language. "
    "Non-English reviews will be translated to English before "
    "DistilBERT determines whether the sentiment is Positive or Negative."
)


# =========================================================
# ABOUT MODEL
# =========================================================

with st.expander(
    "🤖 About the Model",
    expanded=False
):

    st.markdown(
        "### DistilBERT"
    )

    st.write(
        "**Task:** Binary movie-review sentiment classification"
    )

    st.write(
        "**Process:** Language detection → Translation if needed → "
        "Text preprocessing → Tokenization → Fine-tuned DistilBERT"
    )

    st.write(
        "**Maximum sequence length:** 256 tokens"
    )

    st.write(
        "**Accuracy:** 90.76%"
    )

    st.write(
        "DistilBERT was selected for the prototype because it achieved "
        "the strongest classification performance during model evaluation."
    )

    st.info(
        "ℹ️ This prototype performs binary sentiment classification only: "
        "Positive or Negative. Neutral or mixed comments may therefore be "
        "less accurately represented because Neutral is not a separate class. "
        "Non-English reviews are translated to English before classification, "
        "so translation may slightly affect the original wording or sentiment."
    )


# =========================================================
# MODEL PERFORMANCE
# =========================================================

with st.expander(
    "📊 Model Performance",
    expanded=False
):

    performance_columns = st.columns(4)

    for column, (metric_name, value) in zip(
        performance_columns,
        BERT_METRICS.items()
    ):

        with column:
            st.metric(
                metric_name,
                f"{value * 100:.2f}%"
            )


# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:

    st.header(
        "ℹ️ About"
    )

    st.write(
        "This application uses a fine-tuned **DistilBERT** model "
        "to classify movie reviews as **Positive** or **Negative**."
    )

    st.write(
        "Non-English reviews are automatically translated "
        "to English before classification."
    )

    st.write(
        f"Accepted review length: "
        f"**{MIN_CHARS}–{MAX_CHARS} characters**."
    )

    st.caption(
        "⚠️ Binary classification only: neutral or mixed comments "
        "may be less accurately represented because the model predicts "
        "only Positive or Negative."
    )

    st.divider()

    st.subheader(
        "Try an example"
    )

    example_reviews = {

        "😍 Positive review":
            "This film was an absolute masterpiece. "
            "The acting, the score, and the cinematography "
            "were excellent. I loved every moment.",

        "🤢 Negative review":
            "What a waste of time. "
            "The plot made no sense, "
            "the dialogue was terrible, "
            "and the movie was extremely boring."
    }

    for label, text in example_reviews.items():

        if st.button(
            label,
            use_container_width=True
        ):
            st.session_state.example_text = text
            st.rerun()

    if st.session_state.history:

        st.divider()

        st.subheader(
            "📜 Session history"
        )

        st.caption(
            f"{len(st.session_state.history)} "
            "review(s) analyzed"
        )

        if st.button(
            "Clear history",
            use_container_width=True
        ):
            st.session_state.history = []
            st.rerun()


# =========================================================
# USER INPUT
# =========================================================

review_text = st.text_area(

    "Your movie review:",

    value=st.session_state.example_text,

    height=150,

    placeholder=(
        "Type or paste a movie review here..."
    )
)

st.caption(
    f"{len(review_text)} / {MAX_CHARS} characters"
)

st.caption(
    "🌐 Non-English reviews will be automatically "
    "translated to English before analysis."
)


# =========================================================
# RESULT CARD
# =========================================================

def render_result_card(
    label,
    confidence
):

    css_class = (
        "result-positive"
        if label == "Positive"
        else "result-negative"
    )

    icon = (
        "😊"
        if label == "Positive"
        else "☹️"
    )

    html = (
        f'<div class="result-card {css_class}">'
        f'<div class="result-label">'
        f'{icon} {label}'
        f'</div>'
        f'<div class="result-sub">'
        f'DistilBERT · Confidence '
        f'{confidence * 100:.1f}%'
        f'</div>'
        f'</div>'
    )

    st.markdown(
        html,
        unsafe_allow_html=True
    )

    st.progress(
        min(
            max(confidence, 0.0),
            1.0
        )
    )

    st.caption(
        "Confidence represents the model's prediction probability "
        "and does not guarantee that the prediction is correct."
    )


# =========================================================
# ANALYZE SENTIMENT
# =========================================================

if st.button(
    "Analyze Sentiment",
    type="primary"
):

    is_valid, error_message = (
        validate_review(review_text)
    )

    if not is_valid:

        st.warning(
            error_message
        )

    else:

        try:

            with st.spinner(
                "Detecting language and analyzing sentiment..."
            ):

                (
                    analysis_text,
                    detected_language,
                    was_translated
                ) = detect_and_translate(
                    review_text
                )

                # Show translation information
                if was_translated:

                    st.info(
                        f"🌐 Detected language: "
                        f"`{detected_language}`. "
                        f"The review was translated to English "
                        f"before sentiment analysis."
                    )

                    st.markdown(
                        "**Translated English review:**"
                    )

                    st.write(
                        analysis_text
                    )

                else:

                    st.caption(
                        "🌐 Detected language: English"
                    )

                # Preprocess text for DistilBERT
                bert_text = (
                    preprocess_bert(
                        analysis_text
                    )
                )

                # Tokenization
                inputs = (
                    bert_tokenizer(
                        bert_text,
                        truncation=True,
                        padding=True,
                        max_length=MAX_LEN,
                        return_tensors="pt"
                    )
                )

                # Prediction
                with torch.no_grad():

                    outputs = (
                        bert_model(
                            **inputs
                        )
                    )

                    probabilities = (
                        torch.softmax(
                            outputs.logits,
                            dim=1
                        )[0]
                    )

                    prediction = (
                        torch.argmax(
                            probabilities
                        ).item()
                    )

                confidence = (
                    torch.max(
                        probabilities
                    ).item()
                )

                label = (
                    "Positive"
                    if prediction == 1
                    else "Negative"
                )

            # Display final result
            st.subheader(
                "Sentiment Result"
            )

            render_result_card(
                label,
                confidence
            )

            # Extra warning for uncertain predictions
            if confidence < 0.60:

                st.warning(
                    "The model has relatively low confidence "
                    "in this prediction. The review may contain "
                    "unclear, ambiguous, neutral, or mixed sentiment."
                )

            # Save to session history
            st.session_state.history.append({

                "Review":
                    review_text.strip()[:70]
                    + (
                        "..."
                        if len(
                            review_text.strip()
                        ) > 70
                        else ""
                    ),

                "Language":
                    detected_language,

                "Translated":
                    "Yes"
                    if was_translated
                    else "No",

                "Prediction":
                    label,

                "Confidence":
                    f"{confidence * 100:.1f}%"
            })

        except Exception as error:

            st.error(
                str(error)
            )


# =========================================================
# SESSION HISTORY TABLE
# =========================================================

if st.session_state.history:

    st.divider()

    st.subheader(
        "📜 Analysis History "
        "(This Session)"
    )

    st.dataframe(
        st.session_state.history,
        use_container_width=True,
        hide_index=True
    )
