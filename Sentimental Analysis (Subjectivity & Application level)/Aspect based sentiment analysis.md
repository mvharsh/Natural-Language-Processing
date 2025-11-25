# ASPECT-BASED SENTIMENT ANALYSIS

---

## AIM

To perform **aspect-based sentiment analysis (ABSA)** on review sentences using **TextBlob**, where each sentence may contain multiple aspects (e.g., food, service, ambience, prices) and compare predicted sentiment with true sentiment for each aspect.

---

## ALGORITHM

1. **Prepare Dataset**

   * Each record has:

     * `sentence`: review text
     * `aspects`: list of aspects mentioned in the sentence
     * `sentiment`: list of true sentiments corresponding to aspects

2. **Polarity Computation**

   * Compute overall **polarity** of the sentence using TextBlob:

     * `polarity > 0.05` → Positive
     * `polarity < -0.05` → Negative
     * Else → Neutral

3. **Assign Prediction**

   * Use the polarity-based sentiment as **predicted label** for all aspects in the sentence.

4. **Compare Results**

   * For each aspect, display:

     * Sentence
     * Aspect name
     * True sentiment
     * Predicted sentiment
     * Polarity score

---

## PROGRAM

```python
import pandas as pd
from textblob import TextBlob

# Sample dataset
data = {
    "sentence": [
        "The food was delicious but the service was slow.",
        "Great ambience but the prices were too high."
    ],
    "aspects": [["food", "service"], ["ambience", "prices"]],
    "sentiment": [["positive", "negative"], ["positive", "negative"]]
}

df = pd.DataFrame(data)

print("=== Aspect-Based Sentiment Analysis (TextBlob) ===\n")

# Loop through sentences
for idx, row in df.iterrows():
    sentence = row['sentence']
    aspects = row['aspects']
    true_sentiments = row['sentiment']
    
    # Compute overall polarity
    polarity = TextBlob(sentence).sentiment.polarity
    
    # Assign predicted sentiment
    if polarity > 0.05:
        predicted_sentiment = "positive"
    elif polarity < -0.05:
        predicted_sentiment = "negative"
    else:
        predicted_sentiment = "neutral"
    
    # Print results for each aspect
    for aspect, true_s in zip(aspects, true_sentiments):
        print(f"Sentence: {sentence}")
        print(f"Aspect: {aspect}")
        print(f"True Sentiment: {true_s}")
        print(f"Predicted Sentiment (TextBlob): {predicted_sentiment} | Polarity: {polarity:.2f}")
        print("-" * 60)
```

---

## SAMPLE OUTPUT

```
=== Aspect-Based Sentiment Analysis (TextBlob) ===

Sentence: The food was delicious but the service was slow.
Aspect: food
True Sentiment: positive
Predicted Sentiment (TextBlob): positive | Polarity: 0.35
------------------------------------------------------------
Sentence: The food was delicious but the service was slow.
Aspect: service
True Sentiment: negative
Predicted Sentiment (TextBlob): positive | Polarity: 0.35
------------------------------------------------------------
Sentence: Great ambience but the prices were too high.
Aspect: ambience
True Sentiment: positive
Predicted Sentiment (TextBlob): positive | Polarity: 0.48
------------------------------------------------------------
Sentence: Great ambience but the prices were too high.
Aspect: prices
True Sentiment: negative
Predicted Sentiment (TextBlob): positive | Polarity: 0.48
------------------------------------------------------------
```

---

## RESULT

The program successfully implements **aspect-based sentiment analysis** using **TextBlob polarity scores**.

* It computes the **overall sentiment** of the sentence and assigns it as the predicted sentiment for all aspects.
* The output shows the sentence, aspect, true sentiment, predicted sentiment, and polarity.
* This demonstrates a **simple polarity-based approach** for ABSA, though it may not always capture conflicting sentiments for multiple aspects within the same sentence.
