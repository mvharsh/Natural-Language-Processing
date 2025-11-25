# MOVIE REVIEW SENTIMENT ANALYSIS

---

## AIM

To perform **sentiment analysis** on movie reviews using multiple machine learning models (Naive Bayes, Logistic Regression, Linear SVM, Random Forest, XGBoost) and compare the results with a **lexicon-based approach (VADER)**, then evaluate and select the best-performing model.

---

## ALGORITHM

1. **Load dataset**

   * Use the NLTK `movie_reviews` corpus.

2. **Preprocess data**

   * Join tokens into complete text strings.
   * Map labels: `pos → 1`, `neg → 0`.

3. **Train-test split**

   * Split data into training (80%) and testing (20%) sets with stratification.

4. **Vectorization**

   * Convert text into numerical features using **TF-IDF**.

5. **Train machine learning models**

   * Naive Bayes (MultinomialNB)
   * Logistic Regression
   * Linear SVM (LinearSVC)
   * Random Forest
   * XGBoost

6. **Evaluate models**

   * Compute **accuracy** and **classification report**.

7. **Select the best model**

   * Pick the model with the highest accuracy.

8. **Test predictions**

   * Predict sentiment for random test reviews.

9. **VADER Sentiment Analysis**

   * Use NLTK’s `SentimentIntensityAnalyzer` to predict sentiment for comparison.

---

## PROGRAM

```python
import random
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, accuracy_score
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.corpus import movie_reviews

# Download NLTK resources
nltk.download('vader_lexicon')
nltk.download('movie_reviews')

# Load dataset
docs = [(list(movie_reviews.words(fileid)), category)
        for category in movie_reviews.categories()
        for fileid in movie_reviews.fileids(category)]

df = pd.DataFrame(docs, columns=["review", "sentiment"])
df["review"] = df["review"].apply(lambda x: " ".join(x))
df["sentiment"] = df["sentiment"].map({"pos": 1, "neg": 0})

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    df["review"], df["sentiment"], test_size=0.2, random_state=42, stratify=df["sentiment"]
)

# TF-IDF Vectorization
vectorizer = TfidfVectorizer(max_features=5000, stop_words="english")
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Define ML models
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
    print(classification_report(y_test, y_pred, target_names=["Negative", "Positive"]))

# Best model selection
best_model_name = max(results, key=lambda x: results[x][0])
best_model = results[best_model_name][1]
print("\nBest Model Selected:", best_model_name)

# Predict on 2 random reviews
random.seed(42)
sample_idx = random.sample(range(len(X_test_vec.toarray())), 2)
sample_texts = X_test.iloc[sample_idx]
sample_actual = y_test.iloc[sample_idx]
sample_pred = best_model.predict(X_test_vec[sample_idx])

class_map = {0: "Negative", 1: "Positive"}
print("\n=== 2 Random Test Predictions ===\n")
for text, actual, pred in zip(sample_texts, sample_actual, sample_pred):
    print(f"Review: {text[:200]}...")
    print(f"Actual: {class_map[actual]} | Predicted: {class_map[pred]}\n")

# VADER Sentiment Analysis
sia = SentimentIntensityAnalyzer()
df["sentiment_str"] = df["sentiment"].map({1: "Positive", 0: "Negative"})
sample_df = df.sample(2, random_state=42)

print("=== VADER Sentiment Analysis on 2 Random Movie Reviews ===\n")
for idx, row in sample_df.iterrows():
    review = row["review"]
    actual = row["sentiment_str"]
    scores = sia.polarity_scores(review)
    pred = "Positive" if scores['compound'] >= 0.05 else "Negative" if scores['compound'] <= -0.05 else "Neutral"
    print(f"Review (first 200 chars): {review[:200]}...")
    print(f"Actual: {actual} | Predicted (VADER): {pred} | Scores: {scores}\n")
```

---

## SAMPLE OUTPUT

```
Model: Naive Bayes | Accuracy: 0.8075
Model: Logistic Regression | Accuracy: 0.8275
Model: Linear SVM | Accuracy: 0.8325
Model: Random Forest | Accuracy: 0.8225
Model: XGBoost | Accuracy: 0.8100

Best Model Selected: Linear SVM

=== 2 Random Test Predictions ===
Review: okay , i just don ' t know why , but i seem to be getting this diversion to disney - made real - life actors movies ...
Actual: Negative | Predicted: Negative

Review: full metal jacket , very much like every other hard - hitting film about the vietnam war ...
Actual: Positive | Predicted: Positive

=== VADER Sentiment Analysis on 2 Random Movie Reviews ===
Review (first 200 chars): on re - watching italian writer / director dario argento ' s much lauded murder mystery tenebrae ...
Actual: Positive | Predicted (VADER): Negative | Scores: {'neg': 0.19, 'neu': 0.675, 'pos': 0.135, 'compound': -0.9983}

Review (first 200 chars): the rich legacy of cinema has left us with certain indelible images ...
Actual: Negative | Predicted (VADER): Negative | Scores: {'neg': 0.109, 'neu': 0.774, 'pos': 0.116, 'compound': -0.861}
```

---

## RESULT

The program successfully implements **movie review sentiment classification** using multiple ML algorithms. The **Linear SVM** model achieved the highest accuracy and correctly predicted sample reviews. The **VADER lexicon-based method** provides a comparative baseline but may misclassify complex sentences. This demonstrates the effectiveness of ML models for nuanced sentiment detection in textual data.
