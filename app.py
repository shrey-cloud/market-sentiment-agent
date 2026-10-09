import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report

# -------------------------------
# Page Title
# -------------------------------
st.set_page_config(page_title="Market Sentiment Agent", layout="centered")
st.title("📊 Market Sentiment Agent (By Shreya Gaikwad)")
st.write("Analyze financial news headlines and get a Bullish/Bearish/Neutral market signal.")

# -------------------------------
# Load Dataset
# -------------------------------
df = pd.read_csv("financial_phrasebank.csv")

# Map labels
label_map = {"negative": 0, "neutral": 1, "positive": 2}
df["label"] = df["sentiment"].str.lower().map(label_map)

X = df["sentence"].astype(str)
y = df["label"]

# -------------------------------
# Train Model
# -------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

model = Pipeline([
    ("tfidf", TfidfVectorizer(max_features=5000, ngram_range=(1, 2))),
    ("clf", LogisticRegression(max_iter=1000))
])

model.fit(X_train, y_train)

# -------------------------------
# Evaluate on Held-Out Test Split
# -------------------------------
y_pred = model.predict(X_test)
print(f"Test accuracy: {accuracy_score(y_test, y_pred):.3f}")
print(classification_report(y_test, y_pred, target_names=["negative", "neutral", "positive"], digits=3))

label_reverse = {0: "negative", 1: "neutral", 2: "positive"}

# -------------------------------
# User Inputs
# -------------------------------
st.subheader("Enter Stock Information")

ticker = st.text_input("Stock Ticker (e.g., AAPL)", "AAPL")

headlines_input = st.text_area(
    "Paste Headlines (one per line)",
    """Apple reports record quarterly revenue.
Regulators open antitrust investigation into Apple.
Analysts warn about slowing demand.
Apple announces share buybacks."""
)

# -------------------------------
# Run Analysis
# -------------------------------
if st.button("Analyze Sentiment"):
    headlines = [h.strip() for h in headlines_input.split("\n") if h.strip()]

    if len(headlines) == 0:
        st.error("Please enter at least one headline.")
    else:
        preds = model.predict(headlines)
        probs = model.predict_proba(headlines)

        results = []
        counts = {"positive": 0, "neutral": 0, "negative": 0}

        st.subheader("Headline Sentiment Results")

        for text, p, pr in zip(headlines, preds, probs):
            sentiment_label = label_reverse[p]
            counts[sentiment_label] += 1
            st.write(f"**{sentiment_label.upper()}** → {text}")

        pos, neu, neg = counts["positive"], counts["neutral"], counts["negative"]

        # Determine Market Signal
        if pos > neg and pos >= neu:
            signal = "BULLISH 📈"
        elif neg > pos and neg >= neu:
            signal = "BEARISH 📉"
        else:
            signal = "NEUTRAL ➖"

        st.subheader("Final Market Signal")
        st.write(f"**{signal}**")

        st.info(f"Positive: {pos} | Negative: {neg} | Neutral: {neu}")

# -------------------------------
# End of app (NO launch command!)
# -------------------------------
