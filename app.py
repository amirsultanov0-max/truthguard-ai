import re
import json
import joblib
import streamlit as st


# ============================================================
# Page configuration
# ============================================================
st.set_page_config(
    page_title="TruthGuard AI",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# Project paths
# ============================================================
PIPELINE_PATH = "models/fake_news_pipeline.pkl"
METRICS_PATH = "reports/model_metrics.json"


# ============================================================
# Settings
# ============================================================
UNCERTAIN_THRESHOLD = 65  # If confidence is below 65%, show "Uncertain"


# ============================================================
# Text cleaning function
# ============================================================
def clean_text(text):
    """
    Clean raw text before sending it to the trained ML pipeline.
    This function matches the preprocessing used during training.
    """
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", " ", text)
    text = re.sub(r"[^a-zA-Z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


# ============================================================
# Load trained pipeline and metrics
# ============================================================
@st.cache_resource
def load_pipeline():
    return joblib.load(PIPELINE_PATH)


@st.cache_data
def load_metrics():
    try:
        with open(METRICS_PATH, "r", encoding="utf-8") as file:
            return json.load(file)
    except FileNotFoundError:
        return None


pipeline = load_pipeline()
metrics = load_metrics()


# ============================================================
# Custom CSS
# ============================================================
st.markdown(
    """
    <style>
    .main-title {
        font-size: 52px;
        font-weight: 900;
        margin-bottom: 0px;
        color: #111827;
    }

    .subtitle {
        font-size: 20px;
        color: #6b7280;
        margin-top: 8px;
        margin-bottom: 25px;
    }

    .section-card {
        padding: 24px;
        border-radius: 18px;
        border: 1px solid #e5e7eb;
        background-color: #ffffff;
        box-shadow: 0 4px 16px rgba(0,0,0,0.05);
        margin-bottom: 20px;
    }

    .result-card {
        padding: 28px;
        border-radius: 18px;
        margin-top: 24px;
        border: 1px solid #e5e7eb;
        box-shadow: 0 6px 18px rgba(0,0,0,0.08);
    }

    .fake-card {
        background-color: #fff1f2;
        border-left: 9px solid #e11d48;
    }

    .real-card {
        background-color: #ecfdf5;
        border-left: 9px solid #10b981;
    }

    .uncertain-card {
        background-color: #fffbeb;
        border-left: 9px solid #f59e0b;
    }

    .result-title {
        font-size: 30px;
        font-weight: 800;
        margin-bottom: 10px;
    }

    .small-text {
        color: #6b7280;
        font-size: 15px;
        line-height: 1.6;
    }

    .footer-text {
        color: #9ca3af;
        font-size: 13px;
        margin-top: 30px;
    }
    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# Sidebar
# ============================================================
with st.sidebar:
    st.markdown("## 🛡️ TruthGuard AI")
    st.write("A fake news detection system built with NLP and machine learning.")

    st.markdown("---")

    st.markdown("### Project Details")
    st.write("**Task type:** Binary classification")
    st.write("**Input:** News title/article")
    st.write("**Output:** Real, Fake, or Uncertain")
    st.write("**Model:** Logistic Regression")
    st.write("**Feature extraction:** TF-IDF")
    st.write("**Backend:** Scikit-learn Pipeline")

    st.markdown("---")

    st.markdown("### Confidence Rule")
    st.write(
        f"If the model confidence is below **{UNCERTAIN_THRESHOLD}%**, "
        "the result is shown as **Uncertain**."
    )

    st.markdown("---")

    st.markdown("### Important Note")
    st.warning(
        "This tool does not verify facts directly. "
        "It predicts based on text patterns learned from the dataset."
    )


# ============================================================
# Header
# ============================================================
st.markdown('<p class="main-title">🛡️ TruthGuard AI</p>', unsafe_allow_html=True)
st.markdown(
    '<p class="subtitle">Fake News Detection System using Natural Language Processing and Machine Learning</p>',
    unsafe_allow_html=True
)

st.markdown(
    """
    <div class="section-card">
        <b>Purpose:</b> This application analyzes the language patterns of a news headline or article
        and predicts whether it is more likely to be <b>real news</b>, <b>fake news</b>, or
        <b>uncertain</b>.
    </div>
    """,
    unsafe_allow_html=True
)


# ============================================================
# Tabs
# ============================================================
tab1, tab2, tab3, tab4 = st.tabs(
    ["🔍 Detector", "📊 Model Info", "⚙️ How It Works", "ℹ️ About"]
)


# ============================================================
# Detector tab
# ============================================================
with tab1:
    st.subheader("Analyze News Text")

    st.write(
        "Paste a news headline, paragraph, or full article below. "
        "For more reliable results, use at least 20 words."
    )

    user_input = st.text_area(
        "News text:",
        height=260,
        placeholder="Paste a news headline or article here..."
    )

    word_count = len(user_input.split())
    char_count = len(user_input)

    col_words, col_chars = st.columns(2)

    with col_words:
        st.metric("Word Count", word_count)

    with col_chars:
        st.metric("Character Count", char_count)

    analyze_button = st.button("Analyze News", type="primary", use_container_width=True)

    if analyze_button:
        if word_count < 20:
            st.warning(
                "Please enter at least 20 words. Very short text may produce unreliable predictions."
            )
        else:
            cleaned_text = clean_text(user_input)

            prediction = pipeline.predict([cleaned_text])[0]
            probabilities = pipeline.predict_proba([cleaned_text])[0]

            # Label meaning:
            # 0 = Real News
            # 1 = Fake News
            real_probability = probabilities[0] * 100
            fake_probability = probabilities[1] * 100
            confidence = max(real_probability, fake_probability)

            # ------------------------------------------------------------
            # Uncertain result
            # ------------------------------------------------------------
            if confidence < UNCERTAIN_THRESHOLD:
                st.markdown(
                    f"""
                    <div class="result-card uncertain-card">
                        <div class="result-title">⚠️ Prediction: Uncertain</div>
                        <h3>Confidence: {confidence:.2f}%</h3>
                        <p class="small-text">
                            The model is not confident enough to classify this text clearly
                            as real or fake. Manual verification is recommended.
                        </p>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

                st.progress(int(confidence))

                st.markdown("### Result Explanation")
                st.write(
                    "The probabilities are close to each other, so the text does not strongly match "
                    "either the real-news or fake-news patterns learned from the training dataset."
                )

            # ------------------------------------------------------------
            # Fake news result
            # ------------------------------------------------------------
            elif prediction == 1:
                st.markdown(
                    f"""
                    <div class="result-card fake-card">
                        <div class="result-title">🚨 Prediction: Fake News</div>
                        <h3>Confidence: {fake_probability:.2f}%</h3>
                        <p class="small-text">
                            The model detected language patterns that are more similar to fake news
                            articles from the training dataset.
                        </p>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

                st.progress(int(fake_probability))

                st.markdown("### Result Explanation")
                st.write(
                    "The text was classified as **fake news** because its word patterns, phrasing, "
                    "and structure are closer to examples labeled as fake in the training data."
                )

            # ------------------------------------------------------------
            # Real news result
            # ------------------------------------------------------------
            else:
                st.markdown(
                    f"""
                    <div class="result-card real-card">
                        <div class="result-title">✅ Prediction: Real News</div>
                        <h3>Confidence: {real_probability:.2f}%</h3>
                        <p class="small-text">
                            The model detected language patterns that are more similar to real news
                            articles from the training dataset.
                        </p>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

                st.progress(int(real_probability))

                st.markdown("### Result Explanation")
                st.write(
                    "The text was classified as **real news** because its word patterns, phrasing, "
                    "and structure are closer to examples labeled as real in the training data."
                )

            st.markdown("### Probability Breakdown")

            col_real, col_fake = st.columns(2)

            with col_real:
                st.metric("Real News Probability", f"{real_probability:.2f}%")

            with col_fake:
                st.metric("Fake News Probability", f"{fake_probability:.2f}%")

            st.info(
                "Important: This system predicts based on language patterns. "
                "It is not an official fact-checking tool and should not be used as the only source of truth."
            )


# ============================================================
# Model information tab
# ============================================================
with tab2:
    st.subheader("Model Information")

    if metrics:
        accuracy = metrics["accuracy"] * 100
        dataset_rows = metrics["dataset_shape_after_cleaning"]["rows"]
        train_samples = metrics["training_samples"]
        test_samples = metrics["testing_samples"]
    else:
        accuracy = 94.96
        dataset_rows = 63171
        train_samples = 50536
        test_samples = 12635

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Cleaned Dataset Size", f"{dataset_rows:,}")

    with col2:
        st.metric("Accuracy", f"{accuracy:.2f}%")

    with col3:
        st.metric("Training Samples", f"{train_samples:,}")

    with col4:
        st.metric("Testing Samples", f"{test_samples:,}")

    st.markdown("---")

    st.write("### Dataset")
    st.write(
        """
        The system was trained on the WELFake dataset. The dataset contains labeled news articles,
        including both real and fake examples. Each record includes a title, article text, and label.
        """
    )

    st.write("### Label Meaning")
    st.code(
        """
0 = Real News
1 = Fake News
        """
    )

    st.write("### Data Preprocessing")
    st.write(
        """
        Before training, the text data was cleaned and prepared using the following steps:

        - Missing labels were removed
        - Invalid labels were filtered out
        - Title and article text were combined
        - Text was converted to lowercase
        - URLs were removed
        - Punctuation, symbols, and numbers were removed
        - Very short texts were removed
        - Duplicate articles were removed
        """
    )

    st.write("### Feature Extraction")
    st.write(
        """
        The project uses TF-IDF vectorization. TF-IDF converts text into numerical features by
        measuring how important each word or phrase is in a document compared to the full dataset.
        """
    )

    st.write("### Machine Learning Model")
    st.write(
        """
        Logistic Regression was selected because it is efficient, interpretable, and performs well
        on text classification tasks when combined with TF-IDF features.
        """
    )

    st.write("### Evaluation Results")

    if metrics:
        st.code(
            f"""
Accuracy: {metrics["accuracy"]:.4f}

Training samples: {metrics["training_samples"]}
Testing samples: {metrics["testing_samples"]}

Confusion Matrix:
{metrics["confusion_matrix"]}

Label meaning:
0 = Real News
1 = Fake News
            """
        )
    else:
        st.code(
            """
Accuracy: 0.9496

Confusion Matrix:
[[6542, 362],
 [275, 5456]]

Label meaning:
0 = Real News
1 = Fake News
            """
        )


# ============================================================
# How it works tab
# ============================================================
with tab3:
    st.subheader("How the System Works")

    st.markdown(
        """
        <div class="section-card">
            <h3>1. User Input</h3>
            <p class="small-text">
                The user enters a news headline, paragraph, or full article into the application.
            </p>
        </div>

        <div class="section-card">
            <h3>2. Text Cleaning</h3>
            <p class="small-text">
                The system cleans the input by removing URLs, punctuation, numbers, symbols,
                and unnecessary spaces.
            </p>
        </div>

        <div class="section-card">
            <h3>3. TF-IDF Vectorization</h3>
            <p class="small-text">
                The cleaned text is converted into numerical features using the saved TF-IDF vectorizer
                inside the machine learning pipeline.
            </p>
        </div>

        <div class="section-card">
            <h3>4. Model Prediction</h3>
            <p class="small-text">
                The trained Logistic Regression model analyzes the features and predicts probabilities
                for real news and fake news.
            </p>
        </div>

        <div class="section-card">
            <h3>5. Confidence Threshold</h3>
            <p class="small-text">
                If the highest probability is below 65%, the system does not force a final label.
                Instead, it displays the result as uncertain and recommends manual verification.
            </p>
        </div>

        <div class="section-card">
            <h3>6. Result Display</h3>
            <p class="small-text">
                The application shows the prediction, confidence level, probability breakdown,
                and a short explanation.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )


# ============================================================
# About tab
# ============================================================
with tab4:
    st.subheader("About the Project")

    st.write(
        """
        **TruthGuard AI** is a diploma project that demonstrates how machine learning can be used
        to solve a practical real-world problem: fake news detection.
        """
    )

    st.write(
        """
        The project applies Natural Language Processing techniques to process news text and uses a
        supervised machine learning model to classify articles as real, fake, or uncertain.
        """
    )

    st.write("### Practical Importance")
    st.write(
        """
        Fake news can mislead people, spread misinformation, and damage trust in information sources.
        This project shows how AI can help users identify potentially unreliable content.
        """
    )

    st.write("### Why an Uncertain Result Was Added")
    st.write(
        """
        Some texts are short, neutral, or unclear. For these cases, forcing the system to choose
        only real or fake may be misleading. Therefore, the application uses a confidence threshold.
        If the model is not confident enough, the result is shown as uncertain.
        """
    )

    st.write("### Limitations")
    st.warning(
        """
        This system does not perform real fact-checking. It does not search the internet or verify
        whether a statement is true. It only predicts whether a text looks similar to real or fake
        examples from the training dataset.

        Therefore, the result should be treated as a machine learning prediction, not as final proof.
        """
    )

    st.write("### Future Improvements")
    st.write(
        """
        Possible future improvements include:

        - Adding multilingual support
        - Adding source credibility analysis
        - Connecting the system to fact-checking APIs
        - Adding AI-generated image detection
        - Creating a Telegram bot version
        - Storing prediction history in a database
        """
    )

    st.markdown(
        '<p class="footer-text">TruthGuard AI — Fake News Detection System</p>',
        unsafe_allow_html=True
    )