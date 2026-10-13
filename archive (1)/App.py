
import streamlit as st
import pandas as pd

# Import your existing trained model and datasets
from model import model, train_df, val_df, test_df


# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="Customer Intent Classifier",
    page_icon="💬",
    layout="wide"
)

st.title("💬 Customer Service Intent Classification")
st.write(
    "Predict customer intentions using a Naive Bayes "
    "classifier implemented from scratch."
)

st.caption("Algorithm: Naive Bayes | Text classification")


# --------------------------------------------------
# HELPER FUNCTION: EVALUATION METRICS
# --------------------------------------------------

def calculate_metrics(data):
    actual = data["intent"].tolist()
    predicted = model.predict(data["utterance"].tolist())

    total = len(actual)

    if total == 0:
        return 0.0, pd.DataFrame(), pd.DataFrame()

    correct = sum(
        a == p for a, p in zip(actual, predicted)
    )

    accuracy = correct / total

    labels = sorted(set(actual) | set(predicted))
    rows = []

    for label in labels:
        tp = sum(
            a == label and p == label
            for a, p in zip(actual, predicted)
        )

        fp = sum(
            a != label and p == label
            for a, p in zip(actual, predicted)
        )

        fn = sum(
            a == label and p != label
            for a, p in zip(actual, predicted)
        )

        support = sum(a == label for a in actual)

        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0

        f1 = (
            2 * precision * recall / (precision + recall)
            if precision + recall else 0.0
        )

        rows.append({
            "Intent": label,
            "Precision": precision,
            "Recall": recall,
            "F1-score": f1,
            "Support": support
        })

    report = pd.DataFrame(rows)

    matrix = pd.DataFrame(
        0, index=labels, columns=labels, dtype=int
    )

    for a, p in zip(actual, predicted):
        matrix.loc[a, p] += 1

    matrix.index.name = "Actual"
    matrix.columns.name = "Predicted"

    return accuracy, report, matrix


# --------------------------------------------------
# SIDEBAR: DATASET INFORMATION
# --------------------------------------------------

st.sidebar.header("📂 Dataset Information")

st.sidebar.metric("Training samples", len(train_df))
st.sidebar.metric("Validation samples", len(val_df))
st.sidebar.metric("Testing samples", len(test_df))
st.sidebar.metric("Number of intents", len(model.classes))
st.sidebar.metric("Vocabulary size", model.vocab_size)


# --------------------------------------------------
# DASHBOARD METRICS
# --------------------------------------------------

val_accuracy, val_report, val_matrix = calculate_metrics(val_df)
test_accuracy, test_report, test_matrix = calculate_metrics(test_df)

st.subheader("📊 Model Performance")

col1, col2, col3, col4 = st.columns(4)

col1.metric("Validation Accuracy", f"{val_accuracy * 100:.2f}%")
col2.metric("Test Accuracy", f"{test_accuracy * 100:.2f}%")
col3.metric("Total Intents", len(model.classes))
col4.metric("Vocabulary Size", model.vocab_size)

st.divider()


# --------------------------------------------------
# CUSTOMER MESSAGE PREDICTION
# --------------------------------------------------

st.subheader("🔎 Predict Customer Intent")

message = st.text_area(
    "Enter a customer message",
    placeholder="Example: I want to track my order",
    height=100
)

if st.button("Predict Intent", type="primary"):
    if not message.strip():
        st.warning("Please enter a customer message.")
    else:
        prediction = model.predict_one(message)

        st.success("Prediction completed!")
        st.markdown("### Predicted Intent")
        st.info(prediction)

st.divider()


# --------------------------------------------------
# DATASET PREVIEW
# --------------------------------------------------

st.subheader("📁 Dataset Preview")

dataset_choice = st.selectbox(
    "Choose a dataset",
    ["Training", "Validation", "Testing"]
)

dataset_map = {
    "Training": train_df,
    "Validation": val_df,
    "Testing": test_df
}

selected_df = dataset_map[dataset_choice]

st.write("Dataset shape:", selected_df.shape)
st.dataframe(selected_df.head(10), use_container_width=True)


# --------------------------------------------------
# CLASSIFICATION REPORT
# --------------------------------------------------

st.subheader("📋 Classification Report")

report_choice = st.selectbox(
    "Select evaluation dataset",
    ["Validation", "Testing"]
)

if report_choice == "Validation":
    selected_report = val_report
else:
    selected_report = test_report

if not selected_report.empty:
    display_report = selected_report.copy()

    for column in ["Precision", "Recall", "F1-score"]:
        display_report[column] = display_report[column].round(3)

    st.dataframe(
        display_report,
        use_container_width=True,
        hide_index=True
    )

    # Macro average
    macro = selected_report[
        ["Precision", "Recall", "F1-score"]
    ].mean()

    # Weighted average
    support = selected_report["Support"]
    total_support = support.sum()

    if total_support:
        weighted = (
            selected_report[
                ["Precision", "Recall", "F1-score"]
            ].multiply(support, axis=0).sum()
            / total_support
        )

        avg_df = pd.DataFrame([
            {
                "Average": "Macro Average",
                "Precision": macro["Precision"],
                "Recall": macro["Recall"],
                "F1-score": macro["F1-score"]
            },
            {
                "Average": "Weighted Average",
                "Precision": weighted["Precision"],
                "Recall": weighted["Recall"],
                "F1-score": weighted["F1-score"]
            }
        ])

        st.write("**Average Scores**")
        st.dataframe(
            avg_df.round(3),
            use_container_width=True,
            hide_index=True
        )

st.divider()


# --------------------------------------------------
# CONFUSION MATRIX
# --------------------------------------------------

st.subheader("🎯 Confusion Matrix")

st.caption(
    "Rows represent actual intents; columns represent "
    "predicted intents. Diagonal values are correct predictions."
)

if report_choice == "Validation":
    selected_matrix = val_matrix
else:
    selected_matrix = test_matrix

st.dataframe(
    selected_matrix,
    use_container_width=True
)

st.caption(
    "Tip: Large matrices may require horizontal scrolling."
)

st.divider()

st.caption(
    "Customer Service Intent Classification | "
    "Naive Bayes implemented from scratch"
)
