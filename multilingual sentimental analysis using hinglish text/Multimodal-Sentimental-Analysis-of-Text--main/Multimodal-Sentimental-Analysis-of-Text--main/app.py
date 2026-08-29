import streamlit as st
import pandas as pd
import re
import time
from io import StringIO

import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import BernoulliNB, MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from wordcloud import WordCloud

st.set_page_config(page_title="Twitter Sentiment Analysis", layout="wide")

css = '''
<style>
body {background-color: #f6f7fb}
.stApp {padding: 1rem}
.card {background: #ffffff; border-radius: 12px; padding: 18px; box-shadow: 0 6px 18px rgba(0,0,0,0.06)}
.large-text {font-size:20px; font-weight:600}
.result-positive {background:#e6ffed; border-left:6px solid #2ecc71; padding:12px; border-radius:8px}
.result-negative {background:#ffecec; border-left:6px solid #e74c3c; padding:12px; border-radius:8px}
.analyze-btn {background:#4b7bec; color:white}
@media (max-width: 600px) {
  .stSidebar {display: none}
}
</style>
'''

st.markdown(css, unsafe_allow_html=True)

st.title("Twitter Sentiment Analysis using NLP")

st.sidebar.title("Menu")

option = st.sidebar.selectbox(
    "Choose Option",
    ["Single Tweet Analysis", "Dataset Analysis", "Model Performance"],
)


def _safe_text_series(series: pd.Series) -> pd.Series:
    return series.astype(str).fillna("").str.replace(r"\s+", " ", regex=True).str.strip()


def _normalize_binary_labels(raw_labels: pd.Series) -> pd.Series:
    """Return labels as 0/1 for negative/positive where possible."""
    labels = raw_labels.copy()

    numeric_labels = pd.to_numeric(labels, errors="coerce")
    if numeric_labels.notna().all():
        unique_vals = sorted(numeric_labels.unique().tolist())
        if set(unique_vals).issubset({0.0, 1.0}):
            return numeric_labels.astype(int)
        if set(unique_vals).issubset({0.0, 4.0}):
            return (numeric_labels > 0).astype(int)
        if len(unique_vals) == 2:
            min_val = min(unique_vals)
            return (numeric_labels > min_val).astype(int)

    label_text = labels.astype(str).str.lower().str.strip()
    map_dict = {
        "negative": 0,
        "neg": 0,
        "0": 0,
        "positive": 1,
        "pos": 1,
        "1": 1,
        "4": 1,
    }
    mapped = label_text.map(map_dict)
    if mapped.notna().all() and mapped.nunique() == 2:
        return mapped.astype(int)

    raise ValueError("Could not normalize labels to binary 0/1 sentiment values.")


def _prepare_training_data(df: pd.DataFrame):
    text_col_candidates = ["Tweet", "tweet", "text", "review", "content"]
    label_col_candidates = ["Label", "label", "sentiment", "target"]

    text_col = next((col for col in text_col_candidates if col in df.columns), None)
    label_col = next((col for col in label_col_candidates if col in df.columns), None)

    if text_col is None or label_col is None:
        raise ValueError(
            "Dataset must contain a text column (e.g., Tweet) and label column (e.g., Label or sentiment)."
        )

    work_df = df[[text_col, label_col]].dropna().copy()
    work_df[text_col] = _safe_text_series(work_df[text_col])
    work_df = work_df[work_df[text_col] != ""]
    work_df["_label"] = _normalize_binary_labels(work_df[label_col])

    if work_df["_label"].nunique() != 2:
        raise ValueError("The dataset must have exactly two sentiment classes.")

    return work_df, text_col, label_col


def _build_models():
    return {
        "Logistic Regression": LogisticRegression(max_iter=1200, random_state=42),
        "Multinomial Naive Bayes": MultinomialNB(),
        "Bernoulli Naive Bayes": BernoulliNB(),
        "Linear SVM": LinearSVC(random_state=42),
        "SGD Classifier": SGDClassifier(random_state=42),
    }


def _decision_scores(model_pipeline: Pipeline, x_test: pd.Series):
    classifier = model_pipeline.named_steps["classifier"]
    x_vec = model_pipeline.named_steps["tfidf"].transform(x_test)

    if hasattr(classifier, "predict_proba"):
        return classifier.predict_proba(x_vec)[:, 1], True
    if hasattr(classifier, "decision_function"):
        return classifier.decision_function(x_vec), True
    return None, False


@st.cache_data(show_spinner=False)
def _compute_dashboard_artifacts(csv_path: str):
    df = pd.read_csv(csv_path)
    prepared_df, text_col, label_col = _prepare_training_data(df)

    x = prepared_df[text_col]
    y = prepared_df["_label"]

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, stratify=y, random_state=42
    )

    model_results = []
    trained_models = {}

    for model_name, model in _build_models().items():
        pipeline = Pipeline(
            [
                ("tfidf", TfidfVectorizer(stop_words="english", max_features=20000)),
                ("classifier", model),
            ]
        )

        train_start = time.perf_counter()
        pipeline.fit(x_train, y_train)
        train_time = time.perf_counter() - train_start

        pred_start = time.perf_counter()
        y_pred = pipeline.predict(x_test)
        pred_time = time.perf_counter() - pred_start

        model_results.append(
            {
                "Model": model_name,
                "Accuracy": accuracy_score(y_test, y_pred),
                "Precision": precision_score(y_test, y_pred, zero_division=0),
                "Recall": recall_score(y_test, y_pred, zero_division=0),
                "F1 Score": f1_score(y_test, y_pred, zero_division=0),
                "Training Time (s)": train_time,
                "Prediction Time (s)": pred_time,
            }
        )
        trained_models[model_name] = pipeline

    metrics_df = pd.DataFrame(model_results).sort_values(
        by=["F1 Score", "Accuracy"], ascending=False
    )
    best_model_name = metrics_df.iloc[0]["Model"]
    best_model = trained_models[best_model_name]
    best_pred = best_model.predict(x_test)

    cm = confusion_matrix(y_test, best_pred)
    report_dict = classification_report(y_test, best_pred, output_dict=True, zero_division=0)
    report_df = pd.DataFrame(report_dict).transpose()

    roc_payload = None
    scores, has_roc = _decision_scores(best_model, x_test)
    if has_roc and scores is not None:
        fpr, tpr, _ = roc_curve(y_test, scores)
        auc_score = roc_auc_score(y_test, scores)
        roc_payload = {"fpr": fpr, "tpr": tpr, "auc": auc_score}

    tokens = prepared_df[text_col].str.lower().str.findall(r"\b\w+\b")
    vocab = set()
    for token_list in tokens:
        vocab.update(token_list)

    stats = {
        "samples": int(len(prepared_df)),
        "positive": int((prepared_df["_label"] == 1).sum()),
        "negative": int((prepared_df["_label"] == 0).sum()),
        "vocab_size": int(len(vocab)),
        "avg_len": float(tokens.str.len().mean()),
    }

    pos_text = " ".join(prepared_df.loc[prepared_df["_label"] == 1, text_col].astype(str).tolist())
    neg_text = " ".join(prepared_df.loc[prepared_df["_label"] == 0, text_col].astype(str).tolist())

    return {
        "raw_df": df,
        "prepared_df": prepared_df,
        "text_col": text_col,
        "label_col": label_col,
        "x_test": x_test,
        "y_test": y_test,
        "metrics_df": metrics_df,
        "best_model_name": best_model_name,
        "cm": cm,
        "report_df": report_df,
        "roc_payload": roc_payload,
        "stats": stats,
        "pos_text": pos_text,
        "neg_text": neg_text,
    }


def _plot_metric_bar(metrics_df: pd.DataFrame, metric_name: str):
    fig = px.bar(
        metrics_df,
        x="Model",
        y=metric_name,
        color="Model",
        text=metrics_df[metric_name].round(3),
        title=f"{metric_name} Comparison Across Models",
    )
    fig.update_traces(textposition="outside")
    fig.update_layout(showlegend=False, yaxis_range=[0, 1.05], xaxis_title=None)
    return fig


def _render_wordcloud(text: str, title: str):
    if not text.strip():
        st.info(f"No text available for {title} word cloud.")
        return

    wc = WordCloud(width=900, height=450, background_color="white", colormap="viridis").generate(text)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(wc, interpolation="bilinear")
    ax.axis("off")
    ax.set_title(title)
    st.pyplot(fig)

def load_positive_words(path="adjectives.txt"):
    try:
        with open(path, encoding="utf-8") as f:
            words = {w.strip().lower() for w in re.split(r"\s+", f.read()) if w.strip()}
            return words
    except Exception:
        return set()

positive_words = load_positive_words()

with st.container():
    if option == "Single Tweet Analysis":
        st.subheader("Analyze a Tweet")
        left, right = st.columns([2,1])
        with left:
            tweet = st.text_area("Enter Tweet", height=150)
            if st.button("Analyze Sentiment"):
                if tweet.strip() == "":
                    st.warning("Please enter a tweet")
                else:
                    tokens = [t for t in re.findall(r"\w+", tweet.lower())]
                    score = sum(1 for t in tokens if t in positive_words)
                    if score > 0:
                        st.markdown('<div class="card result-positive">Positive Sentiment 😊</div>', unsafe_allow_html=True)
                        st.balloons()
                    else:
                        st.markdown('<div class="card result-negative">Negative Sentiment 😡</div>', unsafe_allow_html=True)
        with right:
            st.markdown("<div class='card'><div class='large-text'>Quick Tips</div><ul><li>Use short, clear tweets</li><li>Add positive keywords like 'good' or 'happy'</li><li>Add translations if using other languages</li></ul></div>", unsafe_allow_html=True)

    elif option == "Dataset Analysis":
        st.subheader("Dataset Sentiment Analysis")
        try:
            data = pd.read_csv("test_data.csv")
        except Exception as e:
            st.error(f"Could not read test_data.csv: {e}")
            data = None

        if data is not None:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.subheader("Dataset Preview")
            st.dataframe(data.head())
            st.write("Total Tweets:", len(data))
            if "sentiment" in data.columns:
                sentiment_counts = data["sentiment"].value_counts()
                st.bar_chart(sentiment_counts)
            st.markdown('</div>', unsafe_allow_html=True)

    elif option == "Model Performance":
        st.subheader("Model Performance Dashboard")
        st.caption("Placement-ready benchmark view for classical NLP sentiment models")

        with st.spinner("Training and evaluating models..."):
            try:
                artifacts = _compute_dashboard_artifacts("test_data.csv")
            except Exception as e:
                st.error(f"Could not generate model performance dashboard: {e}")
                artifacts = None

        if artifacts is not None:
            stats = artifacts["stats"]
            metrics_df = artifacts["metrics_df"].copy()

            st.markdown("<div class='card'>", unsafe_allow_html=True)
            st.markdown("### Dataset Statistics")
            c1, c2, c3, c4, c5 = st.columns(5)
            c1.metric("Number of Samples", f"{stats['samples']:,}")
            c2.metric("Positive Reviews", f"{stats['positive']:,}")
            c3.metric("Negative Reviews", f"{stats['negative']:,}")
            c4.metric("Vocabulary Size", f"{stats['vocab_size']:,}")
            c5.metric("Avg Review Length", f"{stats['avg_len']:.2f} words")
            st.caption(
                "Interview talking point: These stats describe data balance and lexical richness, "
                "which directly affect generalization and model bias."
            )
            st.markdown("</div>", unsafe_allow_html=True)

            st.markdown("<div class='card'>", unsafe_allow_html=True)
            st.markdown("### Model Comparison Table")
            display_df = metrics_df.copy()
            for col in ["Accuracy", "Precision", "Recall", "F1 Score"]:
                display_df[col] = display_df[col].round(4)
            for col in ["Training Time (s)", "Prediction Time (s)"]:
                display_df[col] = display_df[col].round(4)
            st.dataframe(display_df, use_container_width=True)

            csv_buffer = StringIO()
            metrics_df.to_csv(csv_buffer, index=False)
            st.download_button(
                label="Download Model Metrics CSV",
                data=csv_buffer.getvalue(),
                file_name="model_performance_metrics.csv",
                mime="text/csv",
            )
            st.caption(
                "Interview talking point: Accuracy can be misleading on imbalanced data. "
                "Precision/Recall/F1 together provide a better business-risk view."
            )
            st.markdown("</div>", unsafe_allow_html=True)

            st.markdown("<div class='card'>", unsafe_allow_html=True)
            st.markdown("### Sentiment Class Distribution")
            pie_df = pd.DataFrame(
                {
                    "Sentiment": ["Positive", "Negative"],
                    "Count": [stats["positive"], stats["negative"]],
                }
            )
            pie_fig = px.pie(
                pie_df,
                values="Count",
                names="Sentiment",
                color="Sentiment",
                color_discrete_map={"Positive": "#2ecc71", "Negative": "#e74c3c"},
                title="Positive vs Negative Distribution",
            )
            st.plotly_chart(pie_fig, use_container_width=True)
            st.caption(
                "Interview talking point: This chart highlights class imbalance; if one class dominates, "
                "use stratified split and class-aware metrics to avoid over-optimistic evaluation."
            )
            st.markdown("</div>", unsafe_allow_html=True)

            st.markdown("<div class='card'>", unsafe_allow_html=True)
            st.markdown("### Metric Comparison Graphs")
            st.plotly_chart(_plot_metric_bar(metrics_df, "Accuracy"), use_container_width=True)
            st.caption(
                "Accuracy comparison: overall correctness. Useful for quick benchmarking but not sufficient "
                "when class distribution is uneven."
            )
            st.plotly_chart(_plot_metric_bar(metrics_df, "Precision"), use_container_width=True)
            st.caption(
                "Precision comparison: among predicted positive tweets, how many are truly positive. "
                "Higher precision means fewer false positives."
            )
            st.plotly_chart(_plot_metric_bar(metrics_df, "Recall"), use_container_width=True)
            st.caption(
                "Recall comparison: among truly positive tweets, how many are captured by the model. "
                "Higher recall means fewer false negatives."
            )
            st.plotly_chart(_plot_metric_bar(metrics_df, "F1 Score"), use_container_width=True)
            st.caption(
                "F1 comparison: harmonic mean of precision and recall, ideal when both false positives and "
                "false negatives matter."
            )
            st.markdown("</div>", unsafe_allow_html=True)

            st.markdown("<div class='card'>", unsafe_allow_html=True)
            st.markdown(f"### Confusion Matrix (Best Model: {artifacts['best_model_name']})")
            cm = artifacts["cm"]
            cm_fig = px.imshow(
                cm,
                text_auto=True,
                color_continuous_scale="Blues",
                labels=dict(x="Predicted Label", y="True Label", color="Count"),
                x=["Negative", "Positive"],
                y=["Negative", "Positive"],
            )
            st.plotly_chart(cm_fig, use_container_width=True)
            st.caption(
                "Interview talking point: The confusion matrix pinpoints error types. "
                "Top-right are false positives; bottom-left are false negatives."
            )
            st.markdown("</div>", unsafe_allow_html=True)

            st.markdown("<div class='card'>", unsafe_allow_html=True)
            st.markdown("### Classification Report")
            report_df = artifacts["report_df"].copy()
            numeric_cols = report_df.select_dtypes(include="number").columns
            report_df[numeric_cols] = report_df[numeric_cols].round(4)
            st.dataframe(report_df, use_container_width=True)
            st.caption(
                "Interview talking point: Support indicates sample count per class. "
                "Macro average treats classes equally; weighted average reflects class frequency."
            )
            st.markdown("</div>", unsafe_allow_html=True)

            st.markdown("<div class='card'>", unsafe_allow_html=True)
            st.markdown("### ROC Curve")
            roc_payload = artifacts["roc_payload"]
            if roc_payload is None:
                st.info("ROC curve is not available for the selected best model output type.")
            else:
                roc_fig = go.Figure()
                roc_fig.add_trace(
                    go.Scatter(
                        x=roc_payload["fpr"],
                        y=roc_payload["tpr"],
                        mode="lines",
                        name=f"ROC (AUC = {roc_payload['auc']:.4f})",
                        line=dict(width=3),
                    )
                )
                roc_fig.add_trace(
                    go.Scatter(
                        x=[0, 1],
                        y=[0, 1],
                        mode="lines",
                        name="Random Baseline",
                        line=dict(dash="dash"),
                    )
                )
                roc_fig.update_layout(
                    title="ROC Curve for Best-Performing Model",
                    xaxis_title="False Positive Rate",
                    yaxis_title="True Positive Rate",
                )
                st.plotly_chart(roc_fig, use_container_width=True)
                st.caption(
                    "Interview talking point: ROC-AUC summarizes separability across thresholds. "
                    "AUC near 1.0 indicates strong class discrimination; 0.5 is random guessing."
                )
            st.markdown("</div>", unsafe_allow_html=True)

            st.markdown("<div class='card'>", unsafe_allow_html=True)
            st.markdown("### Word Clouds by Sentiment")
            left_wc, right_wc = st.columns(2)
            with left_wc:
                _render_wordcloud(artifacts["pos_text"], "Positive Reviews Word Cloud")
                st.caption(
                    "Interview talking point: Frequently appearing positive terms reveal dominant semantic patterns "
                    "the model may rely on."
                )
            with right_wc:
                _render_wordcloud(artifacts["neg_text"], "Negative Reviews Word Cloud")
                st.caption(
                    "Interview talking point: Negative word clusters help explain misclassification risks and "
                    "guide feature engineering."
                )
            st.markdown("</div>", unsafe_allow_html=True)

st.sidebar.subheader("Sample Cleaned Tweets")
tweets = []
for fname in ("cleaned_tweets.txt", "cleaned_tweet.txt"):
    try:
        with open(fname, encoding="utf-8") as f:
            tweets = f.readlines()
            break
    except Exception:
        continue

if tweets:
    st.sidebar.write([t.strip() for t in tweets[:5]])
else:
    st.sidebar.write("No cleaned tweets file found")