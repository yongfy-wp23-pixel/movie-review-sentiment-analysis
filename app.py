import re
from pathlib import Path

import pandas as pd
import streamlit as st
import torch

from transformers import (
    DistilBertTokenizerFast,
    DistilBertForSequenceClassification
)

from langdetect import detect, LangDetectException
from deep_translator import GoogleTranslator


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

MIN_CHARS = 5
MAX_CHARS = 1500
MIN_ALPHA_RATIO = 0.5
MAX_LEN = 256

MOVIES = [
    "Interstellar",
    "The Dark Knight",
    "Titanic",
    "Avengers: Endgame"
]


# ============================================================
# STREAMLIT PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Movie Audience Sentiment Dashboard",
    page_icon="🎬",
    layout="wide"
)


# ============================================================
# CUSTOM STYLE
# ============================================================

st.markdown(
    """
    <style>

    .main-title {
        font-size: 38px;
        font-weight: 700;
        margin-bottom: 5px;
    }

    .subtitle {
        font-size: 17px;
        color: gray;
        margin-bottom: 25px;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# TEXT PREPROCESSING
# ============================================================

def preprocess(text):

    text = text.lower()

    # Remove HTML tags
    text = re.sub(r"<.*?>", " ", text)

    # Remove URLs
    text = re.sub(r"http\S+|www\S+", " ", text)

    # Remove extra spaces
    text = re.sub(r"\s+", " ", text).strip()

    return text


# ============================================================
# LANGUAGE DETECTION
# ============================================================

def detect_language(text):

    # Japanese
    if re.search(r"[\u3040-\u30ff]", text):
        return "ja"

    # Korean
    if re.search(r"[\uac00-\ud7af]", text):
        return "ko"

    # Chinese
    if re.search(r"[\u4e00-\u9fff]", text):
        return "zh-cn"

    try:
        return detect(text)

    except LangDetectException:
        return "unknown"


def language_name(code):

    names = {
        "en": "English",
        "zh-cn": "Chinese",
        "zh-tw": "Chinese",
        "ja": "Japanese",
        "ko": "Korean",
        "ms": "Malay",
        "id": "Indonesian",
        "th": "Thai",
        "vi": "Vietnamese",
        "fr": "French",
        "de": "German",
        "es": "Spanish",
        "it": "Italian",
        "pt": "Portuguese",
        "ru": "Russian",
        "ar": "Arabic",
        "hi": "Hindi",
        "tl": "Filipino",
        "unknown": "Unknown"
    }

    return names.get(code, code.upper())


def detect_and_translate(text):

    lang = detect_language(text)

    if lang == "en":
        return text, lang, False

    if lang == "unknown":
        raise ValueError(
            "Unable to detect the language."
        )

    translated = GoogleTranslator(
        source="auto",
        target="en"
    ).translate(text)

    return translated, lang, True


# ============================================================
# LOAD DISTILBERT MODEL
# ============================================================

@st.cache_resource
def load_model():

    model_dir = BASE_DIR / "distilbert_model"

    tokenizer = DistilBertTokenizerFast.from_pretrained(
        str(model_dir),
        local_files_only=True
    )

    model = DistilBertForSequenceClassification.from_pretrained(
        str(model_dir),
        local_files_only=True
    )

    model.eval()

    return tokenizer, model


# ============================================================
# REVIEW VALIDATION
# ============================================================

def validate_review(text):

    text = text.strip()

    if not text:
        return False, "Please enter a movie review."

    if len(text) < MIN_CHARS:
        return False, (
            f"Please enter at least {MIN_CHARS} characters."
        )

    if len(text) > MAX_CHARS:
        return False, (
            f"Please keep the review under "
            f"{MAX_CHARS} characters."
        )

    letters = sum(
        c.isalpha()
        for c in text
    )

    non_space = sum(
        not c.isspace()
        for c in text
    )

    if (
        non_space == 0
        or letters / non_space < MIN_ALPHA_RATIO
    ):
        return False, (
            "Please enter a meaningful movie review."
        )

    return True, ""


# ============================================================
# SENTIMENT PREDICTION
# ============================================================

def predict_sentiment(review):

    # Load model only when prediction is needed
    tokenizer, model = load_model()

    # Detect and translate language
    translated_text, lang, translated = (
        detect_and_translate(review)
    )

    # Preprocess review
    processed_text = preprocess(
        translated_text
    )

    # Tokenize review
    inputs = tokenizer(
        processed_text,
        truncation=True,
        padding=True,
        max_length=MAX_LEN,
        return_tensors="pt"
    )

    # Perform prediction
    with torch.no_grad():

        outputs = model(**inputs)

        probabilities = torch.softmax(
            outputs.logits,
            dim=1
        )[0]

    prediction = torch.argmax(
        probabilities
    ).item()

    confidence = float(
        torch.max(probabilities)
    )

    label = (
        "Positive"
        if prediction == 1
        else "Negative"
    )

    return {
        "label": label,
        "confidence": confidence,
        "language": language_name(lang),
        "translated": translated,
        "translated_text": translated_text
    }


# ============================================================
# SESSION STATE
# ============================================================

if "reviews" not in st.session_state:
    st.session_state.reviews = []


# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<div class="main-title">'
    '🎬 Movie Audience Sentiment Dashboard'
    '</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Collect audience movie reviews and analyze '
    'overall sentiment for each movie.'
    '</div>',
    unsafe_allow_html=True
)


# ============================================================
# MAIN TABS
# ============================================================

review_tab, producer_tab = st.tabs(
    [
        "📝 Submit Review",
        "📊 Producer Dashboard"
    ]
)


# ============================================================
# TAB 1 - AUDIENCE REVIEW SUBMISSION
# ============================================================

with review_tab:

    st.header(
        "📝 Submit Movie Review"
    )

    st.write(
        "Select a movie and submit your review."
    )

    st.divider()

    selected_movie = st.selectbox(
        "🎬 Select Movie:",
        MOVIES
    )

    review = st.text_area(
        "Your Movie Review:",
        height=170,
        placeholder="Enter your opinion about the movie..."
    )

    st.caption(
        f"{len(review)} / {MAX_CHARS} characters"
    )

    st.caption(
        "🌐 Non-English reviews are automatically "
        "translated into English before processing."
    )


    # ========================================================
    # SUBMIT REVIEW
    # ========================================================

    if st.button(
        "Submit Review",
        type="primary",
        use_container_width=True
    ):

        valid, message = validate_review(
            review
        )

        if not valid:

            st.warning(
                message
            )

        else:

            try:

                with st.spinner(
                    "Submitting review..."
                ):

                    result = predict_sentiment(
                        review
                    )

                # Store review and sentiment result
                st.session_state.reviews.append(
                    {
                        "Movie": selected_movie,
                        "Review": review,
                        "Language": result["language"],
                        "Translated": (
                            "Yes"
                            if result["translated"]
                            else "No"
                        ),
                        "Prediction": result["label"],
                        "Confidence": (
                            result["confidence"] * 100
                        )
                    }
                )

                # Audience only sees submission confirmation
                st.success(
                    "✅ Thank you. "
                    "Your review has been submitted successfully."
                )

                if result["translated"]:

                    st.info(
                        f"🌐 Detected language: "
                        f"{result['language']}. "
                        f"Your review was translated "
                        f"before processing."
                    )

            except Exception as e:

                st.error(
                    "Unable to submit the review."
                )

                st.exception(e)


# ============================================================
# TAB 2 - PRODUCER DASHBOARD
# ============================================================

with producer_tab:

    st.header(
        "📊 Producer Sentiment Dashboard"
    )

    st.write(
        "Select a movie to view its overall "
        "audience sentiment and submitted reviews."
    )

    st.divider()

    producer_movie = st.selectbox(
        "🎬 Select Movie to Analyze:",
        MOVIES,
        key="producer_movie"
    )


    # ========================================================
    # FILTER REVIEWS BY MOVIE
    # ========================================================

    movie_reviews = [
        item
        for item in st.session_state.reviews
        if item["Movie"] == producer_movie
    ]

    total_reviews = len(
        movie_reviews
    )

    positive_reviews = sum(
        item["Prediction"] == "Positive"
        for item in movie_reviews
    )

    negative_reviews = sum(
        item["Prediction"] == "Negative"
        for item in movie_reviews
    )


    # ========================================================
    # CALCULATE PERCENTAGES
    # ========================================================

    if total_reviews > 0:

        positive_rate = (
            positive_reviews
            / total_reviews
            * 100
        )

        negative_rate = (
            negative_reviews
            / total_reviews
            * 100
        )

    else:

        positive_rate = 0
        negative_rate = 0


    # ========================================================
    # SUMMARY METRICS
    # ========================================================

    st.subheader(
        f"🎬 {producer_movie} - Audience Sentiment"
    )

    metric1, metric2, metric3 = st.columns(3)

    metric1.metric(
        "Total Reviews",
        total_reviews
    )

    metric2.metric(
        "😊 Positive Reviews",
        positive_reviews
    )

    metric3.metric(
        "☹️ Negative Reviews",
        negative_reviews
    )


    metric4, metric5 = st.columns(2)

    metric4.metric(
        "Positive Rate",
        f"{positive_rate:.1f}%"
    )

    metric5.metric(
        "Negative Rate",
        f"{negative_rate:.1f}%"
    )


    # ========================================================
    # SENTIMENT DISTRIBUTION
    # ========================================================

    st.divider()

    st.subheader(
        "📈 Sentiment Distribution"
    )

    if total_reviews == 0:

        st.info(
            "No reviews have been submitted "
            "for this movie yet."
        )

    else:

        chart_data = pd.DataFrame(
            {
                "Sentiment": [
                    "Positive",
                    "Negative"
                ],
                "Reviews": [
                    positive_reviews,
                    negative_reviews
                ]
            }
        )

        st.bar_chart(
            chart_data,
            x="Sentiment",
            y="Reviews",
            use_container_width=True
        )


        # ====================================================
        # PRODUCER SUMMARY
        # ====================================================

        st.divider()

        st.subheader(
            "📋 Summary"
        )

        if positive_rate > negative_rate:

            st.success(
                f"Overall audience response is mostly positive. "
                f"{positive_rate:.1f}% of submitted reviews "
                f"for {producer_movie} are positive."
            )

        elif negative_rate > positive_rate:

            st.error(
                f"Overall audience response is mostly negative. "
                f"{negative_rate:.1f}% of submitted reviews "
                f"for {producer_movie} are negative."
            )

        else:

            st.warning(
                "Audience sentiment is evenly divided "
                "between positive and negative reviews."
            )


        # ====================================================
        # REVIEW DETAILS
        # ====================================================

        st.divider()

        st.subheader(
            "💬 Audience Review Details"
        )

        review_dataframe = pd.DataFrame(
            movie_reviews
        )

        review_dataframe["Confidence"] = (
            review_dataframe["Confidence"]
            .apply(
                lambda x: f"{x:.1f}%"
            )
        )

        st.dataframe(
            review_dataframe,
            use_container_width=True,
            hide_index=True
        )


# ============================================================
# OVERALL MOVIE OVERVIEW
# ============================================================

st.divider()

st.header(
    "🎞️ Overall Movie Overview"
)

st.write(
    "Overview of the total number of audience reviews "
    "submitted for each movie."
)

overview_data = []

for movie in MOVIES:

    movie_data = [
        review
        for review in st.session_state.reviews
        if review["Movie"] == movie
    ]

    overview_data.append(
        {
            "Movie": movie,
            "Total Reviews": len(movie_data)
        }
    )


overview_dataframe = pd.DataFrame(
    overview_data
)

st.dataframe(
    overview_dataframe,
    use_container_width=True,
    hide_index=True
)


# ============================================================
# DATA MANAGEMENT
# ============================================================

with st.expander(
    "⚙️ Prototype Data Management"
):

    st.warning(
        "Reviews are stored only during the "
        "current Streamlit session."
    )

    if st.button(
        "Clear All Submitted Reviews"
    ):

        st.session_state.reviews = []

        st.success(
            "All submitted reviews have been cleared."
        )

        st.rerun()