# Market Sentiment Agent

Paste a list of financial news headlines for a stock ticker. The app labels each
headline as positive, neutral, or negative and combines them into one market
signal: **BULLISH**, **BEARISH**, or **NEUTRAL**.

Live demo on Hugging Face Spaces:
https://huggingface.co/spaces/Shreyyycloud16/market-sentiment-agent-shreya
(the Space sleeps when idle, so the first load can take a minute)

Built as the final project for EAI 6050: AI in Finance at Northeastern
University (December 2025).

## How it works

### 1. Classifier: TF-IDF + logistic regression

When the app starts, it trains a scikit-learn `Pipeline` on the Financial
PhraseBank sentences:

- **`TfidfVectorizer(max_features=5000, ngram_range=(1, 2))`** turns each
  sentence into a sparse vector of TF-IDF weights over the 5,000 most frequent
  single words and two-word phrases. Bigrams let the model pick up short
  phrases such as "net loss" or "sales increased", not just single words.
- **`LogisticRegression(max_iter=1000)`** is a multiclass linear classifier
  over those vectors with three labels: negative (0), neutral (1), and
  positive (2).

The data is split 80/20, stratified by label (`random_state=42`), and the model
is fit on the 80% training split. Training takes a few seconds, so there is no
separate saved model file.

### 2. Aggregation into a market signal

Each pasted headline gets a predicted label. The app counts positives (`pos`),
neutrals (`neu`), and negatives (`neg`) and applies these rules:

| Condition | Signal |
|---|---|
| `pos > neg` and `pos >= neu` | BULLISH |
| `neg > pos` and `neg >= neu` | BEARISH |
| anything else (ties, or neutral is the largest group) | NEUTRAL |

The app shows each headline with its label, the final signal, and the
positive/negative/neutral counts.

## Running locally

Requires Python 3.9+.

```bash
git clone https://github.com/shrey-cloud/market-sentiment-agent.git
cd market-sentiment-agent
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Then get the dataset (see below) and start the app:

```bash
streamlit run app.py
```

## Dataset

The model is trained on the **Financial PhraseBank v1.0** (Malo et al., 2013),
using the `Sentences_AllAgree` subset: 2,264 financial news sentences where
every annotator agreed on the label (1,391 neutral, 570 positive, 303
negative).

The dataset is licensed under
[CC BY-NC-SA 3.0](https://creativecommons.org/licenses/by-nc-sa/3.0/) and is
**not included in this repo**. To build the CSV the app expects:

1. Download Financial PhraseBank v1.0. It is distributed by the authors and
   mirrored on the Hugging Face Hub as
   [`takala/financial_phrasebank`](https://huggingface.co/datasets/takala/financial_phrasebank).
2. Convert `Sentences_AllAgree.txt` (one `sentence@label` per line, Latin-1
   encoded) into `financial_phrasebank.csv` with `sentence` and `sentiment`
   columns, in the same folder as `app.py`:

```bash
python - <<'EOF'
import pandas as pd
rows = []
with open("FinancialPhraseBank-v1.0/Sentences_AllAgree.txt", encoding="latin-1") as f:
    for line in f:
        line = line.strip()
        if line:
            sentence, sentiment = line.rsplit("@", 1)
            rows.append((sentence, sentiment))
pd.DataFrame(rows, columns=["sentence", "sentiment"]).to_csv("financial_phrasebank.csv", index=False)
EOF
```

Citation: Malo, P., Sinha, A., Takala, P., Korhonen, P. and Wallenius, J.
(2013). "Good debt or bad debt: Detecting semantic orientations in economic
texts." *Journal of the American Society for Information Science and
Technology*.

## Known limitations

- **Modest accuracy on a small, imbalanced test set.** On the held-out 20%
  split (453 sentences), the model scores 87.9% accuracy and 0.825 macro F1.
  Always predicting neutral would already score 61.4%, so accuracy overstates
  the model's quality. The app prints these metrics to the terminal each time
  the script runs.

  | Class | Precision | Recall | F1 | Test sentences |
  |---|---|---|---|---|
  | negative | 0.929 | 0.639 | 0.757 | 61 |
  | neutral | 0.893 | 0.986 | 0.937 | 278 |
  | positive | 0.817 | 0.746 | 0.780 | 114 |

- **Neutral bias.** About 61% of the training sentences are neutral, so the
  model leans toward neutral, especially on unfamiliar wording. The test
  results confirm it: 98.6% of neutral sentences are caught, but 25% of
  positive sentences are missed, almost all of them (26 of 29) labeled
  neutral.
- **Negatives are often read as positive.** 36% of negative test sentences are
  missed, and most of those (15 of 22) are labeled positive rather than
  neutral. Bad news can therefore push the signal toward BULLISH, and BEARISH
  is the hardest signal to trigger.
- **Vocabulary-bound.** TF-IDF only knows words and phrases seen in about 2,300
  training sentences. Unfamiliar terms, company names, and slang carry no
  signal, and it cannot read negation in context, sarcasm, or how strongly
  something is worded.
- **Domain mismatch.** The training sentences come from formal financial news
  releases, which are written differently from short, punchy headlines.
- **Simple aggregation.** Every headline counts equally, regardless of the
  model's confidence or the headline's importance. The class probabilities are
  computed but not used.
- **No live data.** Headlines are pasted in by hand. The ticker field is not
  used in the analysis or shown in the results, and nothing is fetched for it.
- **Retrains on every interaction.** Streamlit reruns the whole script on each
  button click, so the model is retrained every time (a few seconds on this
  dataset). Caching it with `st.cache_resource`, or training once and loading
  from disk, would fix this.
- **Not a trading signal.** The output describes the tone of the headlines you
  enter. It does not predict price movement.
