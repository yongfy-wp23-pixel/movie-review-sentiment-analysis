import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from pathlib import Path

import streamlit as st
import pickle
import re
import torch

from transformers import (
    DistilBertTokenizerFast,
    DistilBertForSequenceClassification
)

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import nltk


# =========================================================
# PROJECT PATH
# =========================================================

BASE_DIR = Path(__file__).resolve().parent


# =========================================================
# NLTK RESOURCES
# =========================================================

nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)
nltk.download("stopwords", quiet=True)
nltk.download("wordnet", quiet=True)
nltk.download("omw-1.4", quiet=True)

lemmatizer = WordNetLemmatizer()

stop_words = set(stopwords.words("english"))

negation_words = {
    "not",
    "no",
    "never",
    "nobody",
    "none",
    "nothing",
    "neither",
    "nor",
    "nowhere",
    "cannot"
}

stop_words = stop_words - negation_words


# =========================================================
# NAIVE BAYES PREPROCESSING
# =========================================================

def preprocess_text(text):

    text = text.lower()

    # Remove HTML
    text = re.sub(r"<.*?>", " ", text)

    # Keep letters only
    text = re.sub(r"[^a-z\s]", " ", text)

    # Remove extra spaces
    text = re.sub(r"\s+", " ", text).strip()

    # Tokenize
    tokens = word_tokenize(text)

    # Remove stopwords while keeping negations
    tokens = [
        token
        for token in tokens
        if token not in stop_words
    ]

    # Lemmatize
    tokens = [
        lemmatizer.lemmatize(token)
        for token in tokens
    ]

    return " ".join(tokens)


# =========================================================
# DISTILBERT PREPROCESSING
# =========================================================

def preprocess_bert(text):

    text = text.lower()

    # Remove HTML
    text = re.sub(r"<.*?>", " ", text)

    # Remove URLs
    text = re.sub(r"http\S+|www\S+", " ", text)

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
        "we'll": "we will"
    }

    for contraction, expanded in contractions.items():
        text = text.replace(contraction, expanded)

    text = re.sub(r"\s+", " ", text).strip()

    return text


# =========================================================
# BUSINESS RULES
# =========================================================

MIN_CHARS = 10
MAX_CHARS = 3000

MIN_ALPHA_RATIO = 0.5

MIN_CONTRIB_MAGNITUDE = 0.3

MAX_LEN = 256


# =========================================================
# FINAL MODEL PERFORMANCE
# =========================================================

NB_METRICS = {
    "Accuracy": 0.8597,
    "Precision": 0.8644,
    "Recall": 0.8533,
    "F1 Score": 0.8588
}

BERT_METRICS = {
    "Accuracy": 0.9076,
    "Precision": 0.8950,
    "Recall": 0.9235,
    "F1 Score": 0.9090
}


# =========================================================
# LOAD NAIVE BAYES MODEL
# =========================================================

@st.cache_resource
def load_naive_bayes():

    try:

        nb_model_path = (
            BASE_DIR
            / "naive_bayes_model"
            / "nb_model.pkl"
        )

        vectorizer_path = (
            BASE_DIR
            / "naive_bayes_model"
            / "vectorizer.pkl"
        )

        with open(nb_model_path, "rb") as file:
            nb_model = pickle.load(file)

        with open(vectorizer_path, "rb") as file:
            vectorizer = pickle.load(file)

        return nb_model, vectorizer

    except FileNotFoundError as error:

        st.error(
            f"Naive Bayes model files not found: {error}"
        )

        st.stop()

    except Exception as error:

        st.error(
            f"Failed to load Naive Bayes model: {error}"
        )

        st.stop()


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
            f"DistilBERT model not found at "
            f"{model_path}: {error}"
        )

        st.stop()

    except Exception as error:

        st.error(
            f"Failed to load DistilBERT model: "
            f"{error}"
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

    # Minimum characters
    if len(stripped) < MIN_CHARS:

        return (
            False,
            f"Review is too short. "
            f"Please enter at least "
            f"{MIN_CHARS} characters."
        )

    # Maximum characters
    if len(stripped) > MAX_CHARS:

        return (
            False,
            f"Review is too long. "
            f"Please keep it under "
            f"{MAX_CHARS} characters."
        )

    # Check if enough letters exist
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
            "This doesn't look like a text review. "
            "Please enter a real sentence or two."
        )

    # Require at least 3 real words
    words = re.findall(
        r"[a-zA-Z]+",
        stripped
    )

    real_words = [
        word
        for word in words
        if len(word) > 1
    ]

    if len(real_words) < 3:

        return (
            False,
            "Please write a slightly more detailed "
            "review (a few real words)."
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
        font-size: 1.4rem;
        font-weight: 700;
    }

    .result-sub {
        color: rgba(128,128,128,0.9);
        font-size: 0.85rem;
        margin-top: 0.2rem;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# =========================================================
# SESSION HISTORY
# =========================================================

if "history" not in st.session_state:

    st.session_state.history = []


# =========================================================
# LOAD MODELS
# =========================================================

nb_model, vectorizer = load_naive_bayes()

bert_tokenizer, bert_model = load_distilbert()


# =========================================================
# TITLE
# =========================================================

st.title(
    "🎬 Movie Review Sentiment Analysis"
)

st.caption(
    "Enter a movie review below and compare "
    "predictions from Naive Bayes and "
    "DistilBERT side by side."
)


# =========================================================
# MODEL PERFORMANCE SECTION
# =========================================================

with st.expander(
    "📊 Model Performance",
    expanded=False
):

    perf_col1, perf_col2 = st.columns(2)

    with perf_col1:

        st.markdown(
            "**Naive Bayes**"
        )

        for metric_name, value in NB_METRICS.items():

            st.metric(
                metric_name,
                f"{value * 100:.2f}%"
            )

    with perf_col2:

        st.markdown(
            "**DistilBERT**"
        )

        for metric_name, value in BERT_METRICS.items():

            st.metric(
                metric_name,
                f"{value * 100:.2f}%"
            )


# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:

    st.header("ℹ️ About")

    st.write(
        "This application compares a classic "
        "**Naive Bayes** classifier with a "
        "fine-tuned **DistilBERT** transformer "
        "for movie-review sentiment analysis."
    )

    st.write(
        f"Accepted review length: "
        f"**{MIN_CHARS}–{MAX_CHARS} characters**."
    )

    st.divider()

    st.subheader(
        "Try an example"
    )

    example_reviews = {

        "😍 Glowing review":
            "This film was an absolute masterpiece. "
            "The acting, the score, and the cinematography "
            "were excellent. I loved every moment.",

        "🤢 Scathing review":
            "What a waste of time. "
            "The plot made no sense, "
            "the dialogue was terrible, "
            "and the movie was extremely boring.",

        "😐 Mixed review":
            "The visuals were stunning and "
            "the lead performance was solid, "
            "but the pacing dragged badly "
            "and the ending felt rushed."
    }

    for label, text in example_reviews.items():

        if st.button(
            label,
            use_container_width=True
        ):

            st.session_state.example_text = text

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

    value=st.session_state.get(
        "example_text",
        ""
    ),

    height=150,

    placeholder=(
        "Type or paste a movie review here..."
    )
)


# =========================================================
# RESULT CARD
# =========================================================

def render_result_card(model_name, label, confidence, accuracy):

    css_class = "result-positive" if label == "Positive" else "result-negative"
    icon = "😊" if label == "Positive" else "☹️"

    html = (
        f'<div class="result-card {css_class}">'
        f'<div class="result-label">{icon} {label}</div>'
        f'<div class="result-sub">'
        f'{model_name} · Confidence {confidence * 100:.1f}%'
        f'</div>'
        f'</div>'
    )

    st.markdown(
        html,
        unsafe_allow_html=True
    )

    st.progress(
        min(max(confidence, 0.0), 1.0)
    )

    st.caption(
        f"Model accuracy: {accuracy * 100:.2f}%"
    )


# =========================================================
# NAIVE BAYES WORD CONTRIBUTION
# =========================================================

def get_word_contributions(
    cleaned_text,
    vectorizer,
    nb_model
):

    if not hasattr(
        nb_model,
        "feature_log_prob_"
    ):

        return None

    vocab = vectorizer.vocabulary_

    classes = list(
        nb_model.classes_
    )

    if (
        1 not in classes
        or
        0 not in classes
    ):

        return None

    pos_idx = classes.index(1)

    neg_idx = classes.index(0)

    log_prob = (
        nb_model.feature_log_prob_
    )

    contributions = []

    for token in cleaned_text.split():

        if token in vocab:

            feature_index = vocab[token]

            difference = float(
                log_prob[pos_idx, feature_index]
                -
                log_prob[neg_idx, feature_index]
            )

        else:

            difference = 0.0

        contributions.append(
            (token, difference)
        )

    return contributions


# =========================================================
# HIGHLIGHT CONTRIBUTION WORDS
# =========================================================

def render_highlighted_text(contributions):

    if not contributions:
        return "<em>No known words to highlight.</em>"

    max_abs = max(
        (abs(diff) for _, diff in contributions),
        default=0
    ) or 1.0

    spans = []

    for tok, diff in contributions:

        intensity = min(abs(diff) / max_abs, 1.0)

        if diff > 0.05:
            color = f"rgba(46, 204, 113, {0.15 + 0.55 * intensity:.2f})"

        elif diff < -0.05:
            color = f"rgba(231, 76, 60, {0.15 + 0.55 * intensity:.2f})"

        else:
            color = "rgba(128, 128, 128, 0.08)"

        span = (
            f'<span title="log-odds diff: {diff:+.2f}" '
            f'style="background-color:{color}; '
            f'padding:2px 5px; '
            f'border-radius:5px; '
            f'margin:2px 1px; '
            f'display:inline-block; '
            f'line-height:1.9;">'
            f'{tok}'
            f'</span>'
        )

        spans.append(span)

    return " ".join(spans)

# =========================================================
# ANALYZE BUTTON
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

        col1, col2 = st.columns(2)

        pred = None
        pred_bert = None

        nb_confidence = None
        bert_confidence = None


        # =================================================
        # NAIVE BAYES PREDICTION
        # =================================================

        with col1:

            st.subheader(
                "Naive Bayes"
            )

            try:

                with st.spinner(
                    "Running Naive Bayes..."
                ):

                    cleaned = (
                        preprocess_text(
                            review_text
                        )
                    )

                    if not cleaned:

                        st.warning(
                            "After cleaning, "
                            "no meaningful words "
                            "remained in this review."
                        )

                    else:

                        vector = (
                            vectorizer.transform(
                                [cleaned]
                            )
                        )

                        pred = (
                            nb_model.predict(
                                vector
                            )[0]
                        )

                        probability = (
                            nb_model.predict_proba(
                                vector
                            )[0]
                        )

                        nb_confidence = max(
                            probability
                        )

                        label = (
                            "Positive"
                            if pred == 1
                            else "Negative"
                        )

                        render_result_card(
                            "Naive Bayes",
                            label,
                            nb_confidence,
                            NB_METRICS["Accuracy"]
                        )


                        # =============================
                        # WORD EXPLANATION
                        # =============================

                        with st.expander(
                            "🔍 Which words "
                            "influenced this prediction?"
                        ):

                            contributions = (
                                get_word_contributions(
                                    cleaned,
                                    vectorizer,
                                    nb_model
                                )
                            )

                            if contributions is None:

                                st.info(
                                    "This model type "
                                    "doesn't expose "
                                    "per-word "
                                    "log probabilities."
                                )

                            else:

                                st.markdown(
                                    render_highlighted_text(
                                        contributions
                                    ),
                                    unsafe_allow_html=True
                                )

                                st.caption(
                                    "🟩 pushed toward Positive "
                                    "· 🟥 pushed toward Negative "
                                    "· deeper shade = "
                                    "stronger influence"
                                )

                                known = [
                                    contribution
                                    for contribution
                                    in contributions
                                    if contribution[1] != 0.0
                                ]

                                strong = [
                                    contribution
                                    for contribution
                                    in known
                                    if abs(
                                        contribution[1]
                                    )
                                    >=
                                    MIN_CONTRIB_MAGNITUDE
                                ]

                                weak_count = (
                                    len(known)
                                    -
                                    len(strong)
                                )

                                if known:

                                    top_positive = sorted(
                                        [
                                            contribution
                                            for contribution
                                            in strong
                                            if contribution[1] > 0
                                        ],
                                        key=lambda item:
                                            item[1],
                                        reverse=True
                                    )[:5]

                                    top_negative = sorted(
                                        [
                                            contribution
                                            for contribution
                                            in strong
                                            if contribution[1] < 0
                                        ],
                                        key=lambda item:
                                            item[1]
                                    )[:5]

                                    top_col1, top_col2 = (
                                        st.columns(2)
                                    )

                                    with top_col1:

                                        st.markdown(
                                            "**Top Positive words**"
                                        )

                                        if top_positive:

                                            for (
                                                token,
                                                difference
                                            ) in top_positive:

                                                st.write(
                                                    f"🟩 `{token}` "
                                                    f"(+{difference:.2f})"
                                                )

                                        else:

                                            st.caption(
                                                "No words with "
                                                "a strong enough "
                                                "positive signal."
                                            )

                                    with top_col2:

                                        st.markdown(
                                            "**Top Negative words**"
                                        )

                                        if top_negative:

                                            for (
                                                token,
                                                difference
                                            ) in top_negative:

                                                st.write(
                                                    f"🟥 `{token}` "
                                                    f"({difference:.2f})"
                                                )

                                        else:

                                            st.caption(
                                                "No words with "
                                                "a strong enough "
                                                "negative signal."
                                            )

                                    if weak_count:

                                        st.caption(
                                            f"ℹ️ {weak_count} "
                                            "word(s) had a "
                                            "weak/noisy signal "
                                            f"(|diff| < "
                                            f"{MIN_CONTRIB_MAGNITUDE}) "
                                            "and were excluded "
                                            "from these lists."
                                        )

            except Exception as error:

                st.error(
                    "Naive Bayes prediction "
                    f"failed: {error}"
                )


        # =================================================
        # DISTILBERT PREDICTION
        # =================================================

        with col2:

            st.subheader(
                "DistilBERT"
            )

            try:

                with st.spinner(
                    "Running DistilBERT..."
                ):

                    bert_text = (
                        preprocess_bert(
                            review_text
                        )
                    )

                    inputs = (
                        bert_tokenizer(
                            bert_text,
                            truncation=True,
                            padding=True,
                            max_length=MAX_LEN,
                            return_tensors="pt"
                        )
                    )

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

                        pred_bert = (
                            torch.argmax(
                                probabilities
                            ).item()
                        )

                    bert_confidence = (
                        torch.max(
                            probabilities
                        ).item()
                    )

                    label_bert = (
                        "Positive"
                        if pred_bert == 1
                        else "Negative"
                    )

                    render_result_card(
                        "DistilBERT",
                        label_bert,
                        bert_confidence,
                        BERT_METRICS["Accuracy"]
                    )

            except Exception as error:

                st.error(
                    "DistilBERT prediction "
                    f"failed: {error}"
                )


        # =================================================
        # MODEL COMPARISON
        # =================================================

        st.divider()

        if (
            pred is not None
            and
            pred_bert is not None
        ):

            agree = (
                pred == pred_bert
            )

	if agree:
    		st.success("✅ Both models produced the same result!")
	else:
    		st.warning("⚠️ The models produced different results for this review.")

            # =============================================
            # CONFIDENCE COMPARISON
            # =============================================

            st.subheader(
                "Confidence comparison"
            )

            chart_data = {

                "Model": [
                    "Naive Bayes",
                    "DistilBERT"
                ],

                "Confidence (%)": [
                    nb_confidence * 100,
                    bert_confidence * 100
                ]
            }

            st.bar_chart(
                chart_data,
                x="Model",
                y="Confidence (%)",
                horizontal=True
            )


            # =============================================
            # SAVE TO SESSION HISTORY
            # =============================================

            st.session_state.history.append({

                "Review":
                    review_text
                    .strip()[:60]
                    +
                    (
                        "..."
                        if len(
                            review_text.strip()
                        ) > 60
                        else ""
                    ),

                "Naive Bayes":
                    (
                        "Positive"
                        if pred == 1
                        else "Negative"
                    ),

                "NB Confidence":
                    f"{nb_confidence * 100:.1f}%",

                "DistilBERT":
                    (
                        "Positive"
                        if pred_bert == 1
                        else "Negative"
                    ),

                "BERT Confidence":
                    f"{bert_confidence * 100:.1f}%",

                "Result Comparison":
                    (
                        "Same"
                        if agree
                        else "Different"
                    )
            })

        else:

            st.info(
                "Comparison unavailable because "
                "one of the models could not "
                "produce a prediction."
            )


# =========================================================
# SESSION HISTORY TABLE
# =========================================================

if st.session_state.history:

    st.divider()

    st.subheader(
        "📜 Analysis history "
        "(this session)"
    )

    st.dataframe(
        st.session_state.history,
        use_container_width=True,
        hide_index=True
    )