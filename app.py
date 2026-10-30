import re
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st


# --------------------------------------------------
# 1. Page configuration (must be the first Streamlit command)
# --------------------------------------------------
st.set_page_config(
    page_title="AI Sentiment Analysis System",
    page_icon="🎭",
    layout="centered"
)


# --------------------------------------------------
# 2. Preprocessing: keep this identical to the notebook
# --------------------------------------------------
def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


# --------------------------------------------------
# 3. Load the saved TF-IDF + calibrated classifier pipeline
# --------------------------------------------------
@st.cache_resource
def load_model():
    project_root = Path(__file__).resolve().parent
    model_path = project_root / "models" / "sentiment_pipeline.pkl"

    if not model_path.exists():
        raise FileNotFoundError(
            f"Trained model not found at: {model_path}. "
            "Run all notebook cells first to create it."
        )

    return joblib.load(model_path)


# --------------------------------------------------
# 4. User interface
# --------------------------------------------------
st.title("🎭 AI-Powered Sentiment Analysis System")
st.write(
    "Enter a customer review to classify its sentiment as "
    "**Positive, Negative, or Neutral**."
)

try:
    model = load_model()
except Exception as error:
    st.error(f"Unable to load the trained model: {error}")
    st.stop()

user_review = st.text_area(
    "Enter Customer Review Text:",
    placeholder="Example: The product arrived on time and works fantastically well!",
    height=150
)

if st.button("Analyze Sentiment", type="primary", use_container_width=True):
    if not user_review.strip():
        st.warning("Please enter a review before analyzing.")
    else:
        cleaned_review = clean_text(user_review)

        if not cleaned_review:
            st.warning("Please enter text containing English letters.")
        else:
            try:
                prediction = model.predict([cleaned_review])[0]
                probabilities = model.predict_proba([cleaned_review])[0]
                classes = model.classes_

                predicted_probability = float(np.max(probabilities)) * 100

                st.divider()
                st.subheader("Analysis Results")

                if prediction == "Positive":
                    st.success(f"**Predicted Sentiment:** {prediction} 😁")
                elif prediction == "Negative":
                    st.error(f"**Predicted Sentiment:** {prediction} 🙁")
                else:
                    st.info(f"**Predicted Sentiment:** {prediction} 😐")

                st.metric(
                    "Predicted Class Probability",
                    f"{predicted_probability:.2f}%"
                )

                probability_df = pd.DataFrame({
                    "Sentiment Class": classes,
                    "Probability (%)": probabilities * 100
                }).sort_values("Probability (%)", ascending=False)

                st.subheader("Class Probability Breakdown")
                st.bar_chart(
                    probability_df.set_index("Sentiment Class")
                )
                st.dataframe(
                    probability_df.style.format({
                        "Probability (%)": "{:.2f}%"
                    }),
                    use_container_width=True,
                    hide_index=True
                )

                with st.expander("View preprocessed review"):
                    st.write(cleaned_review)

                st.caption(
                    "Probabilities are model estimates and are not a guarantee "
                    "that the prediction is correct."
                )

            except Exception as error:
                st.error(f"Prediction failed: {error}")


# --------------------------------------------------
# 5. Sidebar information
# --------------------------------------------------
st.sidebar.title("About Project")
st.sidebar.info(
    "**Month 2 Project: AI-Powered Sentiment Analysis**\n\n"
    "**Model:** TF-IDF + Calibrated Logistic Regression\n\n"
    "**Original CSV:** 10,000 rows\n\n"
    "**Unique review texts after deduplication:** 300\n\n"
    "**Classes:** Positive, Neutral, Negative\n\n"
    "**Features:** Text preprocessing, TF-IDF bigrams, "
    "sentiment prediction and calibrated probability estimates."
)
