"""
CodeAlpha Task 2: Speech Emotion Recognition from Audio
Train and benchmark Neural MLP, Random Forest, and SVM classifiers on speech acoustic features.
Standalone Model Training & Evaluation CLI.
"""

import os
import argparse
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score, f1_score
import joblib


def main(data_dir=None):
    from ml_engine import ml_engine

    print("======================================================================")
    print("CodeAlpha Task 2: Speech Emotion Recognition Model Training")
    print("======================================================================")

    X = ml_engine.X
    y = ml_engine.y

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    models = {
        "NeuralMLP": Pipeline([
            ("scale", StandardScaler()),
            ("model", MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=600, random_state=42))
        ]),
        "RandomForest": Pipeline([
            ("scale", StandardScaler()),
            ("model", RandomForestClassifier(n_estimators=150, max_depth=10, random_state=42))
        ]),
        "SVM": Pipeline([
            ("scale", StandardScaler()),
            ("model", SVC(kernel="rbf", probability=True, C=2.0, random_state=42))
        ])
    }

    results = []
    best_name, best_pipe, best_acc = None, None, -1.0

    for name, pipe in models.items():
        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)

        results.append({
            "Model": name,
            "Accuracy": round(acc, 4),
            "F1-Score": round(f1, 4)
        })

        print(f"\n--- {name} ---")
        print(classification_report(y_test, y_pred, zero_division=0))
        print(f"Overall Accuracy: {round(acc * 100, 2)}% | Weighted F1: {round(f1, 4)}")

        if acc > best_acc:
            best_name, best_pipe, best_acc = name, pipe, acc

    # Save artifacts
    joblib.dump(best_pipe, "speech_emotion_model.joblib")
    results_df = pd.DataFrame(results)
    results_df.to_csv("model_results.csv", index=False)

    print("\n======================================================================")
    print(f"Model Training Complete! Top Performing Classifier: {best_name} (Accuracy: {round(best_acc * 100, 2)}%)")
    print("Saved Artifacts:")
    print("  -> speech_emotion_model.joblib")
    print("  -> model_results.csv")
    print("======================================================================")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Speech Emotion Recognition Model Trainer")
    parser.add_argument("--data", default=None, help="Directory of audio WAV files (optional)")
    args = parser.parse_args()
    main(args.data)
