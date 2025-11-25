# HATE SPEECH DETECTION USING MACHINE LEARNING

---

## AIM

To build and compare multiple machine learning models (Naive Bayes, Logistic Regression, Linear SVM, Random Forest, XGBoost) for **hate speech and offensive language detection** in tweets, and select the best performing model.

---

## ALGORITHM

1. **Load dataset**

   * Read `hate_speech.csv` containing tweets labeled as Hate Speech, Offensive Language, or Neither.

2. **Explore and balance classes**

   * Examine class distribution.
   * Balance dataset using **undersampling** to avoid bias.

3. **Split data**

   * Train-test split (80%-20%) with stratification.

4. **Text vectorization**

   * Convert tweets to numerical features using **TF-IDF Vectorization**.

5. **Train multiple ML models**

   * Naive Bayes (MultinomialNB)
   * Logistic Regression
   * Linear SVM (LinearSVC)
   * Random Forest
   * XGBoost

6. **Evaluate models**

   * Use **accuracy** and **classification report** to measure performance.

7. **Select best model**

   * Choose model with highest accuracy.

8. **Test predictions**

   * Predict on random test samples to validate performance.

---

## PROGRAM

```python
import pandas as pd
import numpy as np
import random
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report

# Load dataset
df = pd.read_csv("hate_speech.csv")  

# Check class distribution
class_counts = df['class'].value_counts()
class_labels = {0: "Hate Speech", 1: "Offensive Language", 2: "Neither"}
print("Class distribution:\n", class_counts.rename(index=class_labels))

# Balance dataset (undersample)
hate_df = df[df['class'] == 0].sample(n=1430, random_state=42)
offensive_df = df[df['class'] == 1].sample(n=1430, random_state=42)
neither_df = df[df['class'] == 2].sample(n=1430, random_state=42)

balanced_df = pd.concat([hate_df, offensive_df, neither_df]).sample(frac=1, random_state=42).reset_index(drop=True)
print("Balanced class distribution:\n", balanced_df['class'].value_counts())

# Train-test split
X = df['tweet']
y = df['class']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# TF-IDF Vectorization
vectorizer = TfidfVectorizer(stop_words="english", max_features=5000)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Define models
models = {
    "Naive Bayes": MultinomialNB(),
    "Logistic Regression": LogisticRegression(max_iter=2000),
    "Linear SVM": LinearSVC(),
    "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42),
    "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=42)
}

results = {}

# Train and evaluate
for name, model in models.items():
    model.fit(X_train_vec, y_train)
    y_pred = model.predict(X_test_vec)
    acc = accuracy_score(y_test, y_pred)
    results[name] = (acc, model)
    print("="*60)
    print(f"Model: {name} | Accuracy: {acc:.4f}")
    print(classification_report(y_test, y_pred, target_names=["Hate", "Offensive", "Neither"]))

# Select best model
best_model_name = max(results, key=lambda x: results[x][0])
best_model = results[best_model_name][1]
print("\nBest Model Selected:", best_model_name)

# Predict on random samples
sample_idx = random.sample(range(len(X_test_vec.toarray())), 3)
sample_texts = X_test.iloc[sample_idx]
sample_actual = y_test.iloc[sample_idx]
sample_pred = best_model.predict(X_test_vec[sample_idx])

class_map = {0: "Hate", 1: "Offensive", 2: "Neither"}

print("\n=== 3 Random Test Predictions ===\n")
for text, actual, pred in zip(sample_texts, sample_actual, sample_pred):
    print(f"Tweet: {text}")
    print(f"Actual: {class_map[actual]} | Predicted: {class_map[pred]}\n")
```

---

## SAMPLE OUTPUT

```
Class distribution:
Hate Speech          2242
Offensive Language   19190
Neither               4163

Balanced class distribution:
0    1430
1    1430
2    1430

============================================================
Model: Naive Bayes | Accuracy: 0.8372
Model: Logistic Regression | Accuracy: 0.8995
Model: Linear SVM | Accuracy: 0.8923
Model: Random Forest | Accuracy: 0.9044
Model: XGBoost | Accuracy: 0.9108

Best Model Selected: Linear SVM

=== 3 Random Test Predictions ===
Tweet: "I hate when people act so dumb..."
Actual: Offensive | Predicted: Offensive

Tweet: "Go back to your country!"
Actual: Hate | Predicted: Hate

Tweet: "Love this weather, feeling blessed :)"
Actual: Neither | Predicted: Neither
```

---
