# ============================================================
# CUSTOMER SERVICE INTENT CLASSIFICATION
# Using Pandas + Python Built-in Libraries ONLY
# Algorithm: Naive Bayes (implemented from scratch)
# ============================================================

# STEP 1: IMPORT LIBRARIES
# pandas: CSV files, data cleaning and confusion matrix
# re: text tokenization
# math: logarithm for probability calculations
# collections: word and class frequency counting

import pandas as pd
import re
import math
from collections import Counter



# ============================================================
# STEP 2: LOAD DATASETS
# ============================================================

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

train_df = pd.read_csv(
    BASE_DIR / "Bitext_Sample_Customer_Service_Training_Dataset.csv"
)

val_df = pd.read_csv(
    BASE_DIR / "Bitext_Sample_Customer_Service_Validation_Dataset.csv"
)

test_df = pd.read_csv(
    BASE_DIR / "Bitext_Sample_Customer_Service_Testing_Dataset.csv"
)

print("=" * 60)
print("DATASETS LOADED SUCCESSFULLY")
print("=" * 60)

print("Training dataset shape:", train_df.shape)
print("Validation dataset shape:", val_df.shape)
print("Testing dataset shape:", test_df.shape)

print("\nColumn names:", train_df.columns.tolist())



# ============================================================
# STEP 3: DATA CLEANING
# Remove missing and empty messages/intents
# Convert text to lowercase and remove extra spaces
# ============================================================

required_columns = {"utterance", "intent"}

for name, df in [
    ("Training", train_df),
    ("Validation", val_df),
    ("Testing", test_df)
]:
    missing_columns = required_columns - set(df.columns)

    if missing_columns:
        raise ValueError(
            f"{name} dataset is missing columns: {missing_columns}"
        )

    df.dropna(
        subset=["utterance", "intent"],
        inplace=True
    )

    df["utterance"] = (
        df["utterance"].astype(str).str.lower().str.strip()
    )

    df["intent"] = df["intent"].astype(str).str.strip()

    df.drop(
        df[
            (df["utterance"] == "") |
            (df["intent"] == "")
        ].index,
        inplace=True
    )

    df.reset_index(drop=True, inplace=True)

print("\nDATA CLEANING COMPLETED")
print("Training rows after cleaning:", len(train_df))
print("Validation rows after cleaning:", len(val_df))
print("Testing rows after cleaning:", len(test_df))


# ============================================================
# STEP 4: TOKENIZATION
# Split a sentence into individual words
# Example: "Cancel my order" -> ["cancel", "my", "order"]
# ============================================================

def tokenize(text):
    return re.findall(r"[a-z0-9]+", str(text).lower())


# ============================================================
# STEP 5: BUILD NAIVE BAYES CLASSIFIER
# Learn word frequencies for each intent
# ============================================================

class NaiveBayesClassifier:

    def fit(self, data):

        self.class_counts = Counter()
        self.word_counts = {}
        self.total_words = Counter()
        self.vocabulary = set()

        # Learn from each training message
        for _, row in data.iterrows():

            intent = row["intent"]
            words = tokenize(row["utterance"])

            # Count messages belonging to each intent
            self.class_counts[intent] += 1

            # Create a word counter for this intent
            if intent not in self.word_counts:
                self.word_counts[intent] = Counter()

            # Count words appearing in this intent
            self.word_counts[intent].update(words)

            # Count total words for this intent
            self.total_words[intent] += len(words)

            # Store all unique training words
            self.vocabulary.update(words)

        self.total_documents = sum(
            self.class_counts.values()
        )

        self.classes = sorted(self.class_counts.keys())

        self.vocab_size = max(
            len(self.vocabulary), 1
        )

        print("\nMODEL TRAINING COMPLETED")
        print("Total training messages:", self.total_documents)
        print("Total intents:", len(self.classes))
        print("Vocabulary size:", self.vocab_size)

    # ========================================================
    # STEP 6: PREDICT INTENT FOR ONE MESSAGE
    # Use Naive Bayes probability with Laplace smoothing
    # ========================================================

    def predict_one(self, text):

        words = tokenize(text)

        if not words:
            return max(
                self.class_counts,
                key=self.class_counts.get
            )

        best_intent = None
        best_score = float("-inf")

        for intent in self.classes:

            # Prior probability:
            # P(intent) = messages in intent / all messages

            score = math.log(
                self.class_counts[intent] /
                self.total_documents
            )

            # Denominator for smoothed word probability
            denominator = (
                self.total_words[intent] + self.vocab_size
            )

            # Add log probabilities of the message words
            for word in words:

                word_count = self.word_counts[intent].get(
                    word, 0
                )

                probability = (
                    (word_count + 1) / denominator
                )

                score += math.log(probability)

            # Select the intent with the highest score
            if score > best_score:
                best_score = score
                best_intent = intent

        return best_intent

    # ========================================================
    # STEP 7: PREDICT MULTIPLE MESSAGES
    # ========================================================

    def predict(self, texts):

        return [
            self.predict_one(text)
            for text in texts
        ]


# ============================================================
# STEP 8: TRAIN THE MODEL
# Only the training dataset is used for learning
# ============================================================

model = NaiveBayesClassifier()

model.fit(train_df)


# ============================================================
# STEP 9: EVALUATION
# Calculate accuracy, classification report and confusion
# matrix manually using Python and pandas
# ============================================================

def evaluate(data, dataset_name):

    actual = data["intent"].tolist()

    predicted = model.predict(
        data["utterance"].tolist()
    )

    total = len(actual)

    if total == 0:
        print(f"\n{dataset_name}: No samples to evaluate.")
        return 0.0

    correct = sum(
        a == p for a, p in zip(actual, predicted)
    )

    accuracy = correct / total

    print("\n" + "=" * 75)
    print(f"{dataset_name} RESULTS")
    print("=" * 75)

    print("Total samples:", total)
    print("Correct predictions:", correct)
    print("Incorrect predictions:", total - correct)
    print("Accuracy:", round(accuracy * 100, 2), "%")

    # --------------------------------------------------------
    # STEP 9A: CLASSIFICATION REPORT
    # Precision = TP / (TP + FP)
    # Recall    = TP / (TP + FN)
    # F1-score  = 2 * Precision * Recall /
    #             (Precision + Recall)
    # Support   = number of actual samples for the intent
    # --------------------------------------------------------

    labels = sorted(
        set(actual) | set(predicted)
    )

    report_rows = []

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

        support = sum(
            a == label for a in actual
        )

        precision = (
            tp / (tp + fp)
            if tp + fp else 0.0
        )

        recall = (
            tp / (tp + fn)
            if tp + fn else 0.0
        )

        f1_score = (
            2 * precision * recall /
            (precision + recall)
            if precision + recall else 0.0
        )

        report_rows.append({
            "Intent": label,
            "Precision": precision,
            "Recall": recall,
            "F1-score": f1_score,
            "Support": support
        })

    report_df = pd.DataFrame(report_rows)

    print("\nCLASSIFICATION REPORT")
    print("-" * 75)

    print(
        report_df.to_string(
            index=False,
            formatters={
                "Precision": "{:.2f}".format,
                "Recall": "{:.2f}".format,
                "F1-score": "{:.2f}".format
            }
        )
    )

    # --------------------------------------------------------
    # STEP 9B: MACRO AVERAGE
    # Equal weight to each intent
    # --------------------------------------------------------

    macro_precision = report_df["Precision"].mean()
    macro_recall = report_df["Recall"].mean()
    macro_f1 = report_df["F1-score"].mean()

    # --------------------------------------------------------
    # STEP 9C: WEIGHTED AVERAGE
    # Weight each intent by its actual support
    # --------------------------------------------------------

    weighted_precision = (
        (report_df["Precision"] *
         report_df["Support"]).sum() / total
    )

    weighted_recall = (
        (report_df["Recall"] *
         report_df["Support"]).sum() / total
    )

    weighted_f1 = (
        (report_df["F1-score"] *
         report_df["Support"]).sum() / total
    )

    print("\nMACRO AVERAGE")
    print(
        "Precision:", round(macro_precision, 3),
        "| Recall:", round(macro_recall, 3),
        "| F1-score:", round(macro_f1, 3)
    )

    print("\nWEIGHTED AVERAGE")
    print(
        "Precision:", round(weighted_precision, 3),
        "| Recall:", round(weighted_recall, 3),
        "| F1-score:", round(weighted_f1, 3)
    )

    # --------------------------------------------------------
    # STEP 9D: CONFUSION MATRIX
    # Rows = Actual intent
    # Columns = Predicted intent
    # Diagonal values = Correct predictions
    # Off-diagonal values = Incorrect predictions
    # --------------------------------------------------------

    matrix = pd.DataFrame(
        0,
        index=labels,
        columns=labels,
        dtype=int
    )

    for a, p in zip(actual, predicted):
        matrix.loc[a, p] += 1

    matrix.index.name = "Actual"
    matrix.columns.name = "Predicted"

    print("\nCONFUSION MATRIX")
    print("Rows = Actual | Columns = Predicted")
    print(matrix.to_string())

    return accuracy


# ============================================================
# STEP 10: VALIDATION AND TESTING
# Validation: check performance during development
# Testing: final evaluation on the held-out test dataset
# ============================================================

validation_accuracy = evaluate(
    val_df, "VALIDATION"
)

test_accuracy = evaluate(
    test_df, "TEST"
)


# ============================================================
# STEP 11: INFERENCE ON NEW CUSTOMER MESSAGES
# Run this section only when model.py is executed directly
# ============================================================

if __name__ == "__main__":

    print("\n" + "=" * 60)
    print("CUSTOMER MESSAGE INFERENCE")
    print("=" * 60)
    print("Type 'exit' to stop.")

    while True:
        message = input("\nEnter a customer message: ").strip()

        if message.lower() == "exit":
            print("Inference stopped.")
            break

        if not message:
            print("Please enter a non-empty message.")
            continue

        predicted_intent = model.predict_one(message)

        print("Input message:", message)
        print("Predicted intent:", predicted_intent)