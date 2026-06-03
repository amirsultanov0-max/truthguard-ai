import os
import re
import json
import joblib
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)


# ============================================================
# Project paths
# ============================================================
DATA_PATH = "data/WELFake_Dataset.csv"
MODEL_DIR = "models"
REPORT_DIR = "reports"

MODEL_PATH = os.path.join(MODEL_DIR, "fake_news_model.pkl")
VECTORIZER_PATH = os.path.join(MODEL_DIR, "tfidf_vectorizer.pkl")
PIPELINE_PATH = os.path.join(MODEL_DIR, "fake_news_pipeline.pkl")

METRICS_PATH = os.path.join(REPORT_DIR, "model_metrics.json")
REPORT_PATH = os.path.join(REPORT_DIR, "classification_report.txt")
CONFUSION_MATRIX_PATH = os.path.join(REPORT_DIR, "confusion_matrix.png")


# ============================================================
# Create required folders
# ============================================================
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(REPORT_DIR, exist_ok=True)


# ============================================================
# Text cleaning function
# ============================================================
def clean_text(text):
    """
    Cleans raw news text before training/prediction.

    Steps:
    - convert to lowercase
    - remove URLs
    - remove non-English letters, numbers, and symbols
    - remove extra spaces
    """
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", " ", text)
    text = re.sub(r"[^a-zA-Z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


# ============================================================
# Load dataset
# ============================================================
print("=" * 60)
print("Loading dataset...")
print("=" * 60)

df = pd.read_csv(DATA_PATH)

print("Dataset loaded successfully.")
print("Columns:", df.columns.tolist())
print("Original shape:", df.shape)


# ============================================================
# Validate required columns
# ============================================================
required_columns = {"title", "text", "label"}

if not required_columns.issubset(df.columns):
    raise ValueError(
        f"Dataset must contain these columns: {required_columns}. "
        f"Current columns: {df.columns.tolist()}"
    )


# ============================================================
# Basic cleaning
# ============================================================
print("\nCleaning dataset...")

# Keep only useful columns
df = df[["title", "text", "label"]]

# Remove rows with missing labels
df = df.dropna(subset=["label"])

# Make sure labels are integers
df["label"] = df["label"].astype(int)

# Keep only labels 0 and 1
df = df[df["label"].isin([0, 1])]

# Fill missing title/text
df["title"] = df["title"].fillna("")
df["text"] = df["text"].fillna("")

# Combine title and article text
df["content"] = df["title"] + " " + df["text"]

# Clean text
df["content"] = df["content"].apply(clean_text)

# Remove very short texts
df = df[df["content"].str.len() > 30]

# Remove duplicated articles
before_duplicates = df.shape[0]
df = df.drop_duplicates(subset=["content"])
after_duplicates = df.shape[0]

print(f"Removed duplicates: {before_duplicates - after_duplicates}")
print("Final dataset shape:", df.shape)


# ============================================================
# Label information
# ============================================================
print("\nLabel meaning:")
print("0 = Real News")
print("1 = Fake News")

print("\nLabel distribution:")
print(df["label"].value_counts())


# ============================================================
# Features and target
# ============================================================
X = df["content"]
y = df["label"]


# ============================================================
# Train/test split
# ============================================================
print("\nSplitting data into training and testing sets...")

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("Training samples:", X_train.shape[0])
print("Testing samples:", X_test.shape[0])


# ============================================================
# Build ML pipeline
# ============================================================
print("\nBuilding machine learning pipeline...")

pipeline = Pipeline(
    steps=[
        (
            "tfidf",
            TfidfVectorizer(
                max_features=15000,
                stop_words="english",
                ngram_range=(1, 2),
                min_df=2,
                max_df=0.95
            )
        ),
        (
            "model",
            LogisticRegression(
                max_iter=2000,
                class_weight="balanced",
                solver="liblinear",
                random_state=42
            )
        )
    ]
)


# ============================================================
# Train model
# ============================================================
print("\nTraining model...")

pipeline.fit(X_train, y_train)

print("Training completed.")


# ============================================================
# Evaluate model
# ============================================================
print("\nEvaluating model...")

y_pred = pipeline.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
report_text = classification_report(
    y_test,
    y_pred,
    target_names=["Real News", "Fake News"]
)
report_dict = classification_report(
    y_test,
    y_pred,
    target_names=["Real News", "Fake News"],
    output_dict=True
)
cm = confusion_matrix(y_test, y_pred)

print("\nAccuracy:", accuracy)
print("\nClassification Report:")
print(report_text)

print("\nConfusion Matrix:")
print(cm)


# ============================================================
# Save model, vectorizer, and full pipeline
# ============================================================
print("\nSaving model files...")

vectorizer = pipeline.named_steps["tfidf"]
model = pipeline.named_steps["model"]

joblib.dump(model, MODEL_PATH)
joblib.dump(vectorizer, VECTORIZER_PATH)
joblib.dump(pipeline, PIPELINE_PATH)

print(f"Saved model: {MODEL_PATH}")
print(f"Saved vectorizer: {VECTORIZER_PATH}")
print(f"Saved full pipeline: {PIPELINE_PATH}")


# ============================================================
# Save evaluation report
# ============================================================
print("\nSaving evaluation reports...")

with open(REPORT_PATH, "w", encoding="utf-8") as file:
    file.write("TruthGuard AI - Classification Report\n")
    file.write("=" * 50 + "\n\n")
    file.write(f"Accuracy: {accuracy:.4f}\n\n")
    file.write(report_text)
    file.write("\n\nConfusion Matrix:\n")
    file.write(str(cm))
    file.write("\n\nLabel meaning:\n")
    file.write("0 = Real News\n")
    file.write("1 = Fake News\n")

metrics = {
    "accuracy": accuracy,
    "label_meaning": {
        "0": "Real News",
        "1": "Fake News"
    },
    "dataset_shape_after_cleaning": {
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1])
    },
    "training_samples": int(X_train.shape[0]),
    "testing_samples": int(X_test.shape[0]),
    "classification_report": report_dict,
    "confusion_matrix": cm.tolist()
}

with open(METRICS_PATH, "w", encoding="utf-8") as file:
    json.dump(metrics, file, indent=4)

print(f"Saved classification report: {REPORT_PATH}")
print(f"Saved metrics JSON: {METRICS_PATH}")


# ============================================================
# Save confusion matrix image
# ============================================================
display = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["Real News", "Fake News"]
)

display.plot(values_format="d")
plt.title("TruthGuard AI - Confusion Matrix")
plt.tight_layout()
plt.savefig(CONFUSION_MATRIX_PATH, dpi=300)
plt.close()

print(f"Saved confusion matrix image: {CONFUSION_MATRIX_PATH}")


# ============================================================
# Final success message
# ============================================================
print("\n" + "=" * 60)
print("Backend/model training completed successfully.")
print("=" * 60)
print(f"Accuracy: {accuracy:.4f}")
print("Generated files:")
print(f"- {MODEL_PATH}")
print(f"- {VECTORIZER_PATH}")
print(f"- {PIPELINE_PATH}")
print(f"- {REPORT_PATH}")
print(f"- {METRICS_PATH}")
print(f"- {CONFUSION_MATRIX_PATH}")