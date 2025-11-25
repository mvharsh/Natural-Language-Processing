# TEXT REPRESENTATION (FEATURE LEVEL / NUMERICAL REPRESENTATION)

## AIM

To implement Bag of Words (BoW) and TF-IDF (Term Frequency–Inverse Document Frequency) using Scikit-learn in Python for representing text documents numerically.

---

## ALGORITHM

1. **Input Corpus Preparation**

   * Define a small text corpus containing multiple sentences.

2. **Bag of Words (BoW) Representation**

   * Initialize `CountVectorizer()` from Scikit-learn.
   * Fit and transform the corpus to generate the BoW matrix.
   * Convert the resulting sparse matrix into a DataFrame for easy readability.

3. **TF-IDF Representation**

   * Initialize `TfidfVectorizer()` from Scikit-learn.
   * Fit and transform the corpus to generate the TF-IDF matrix.
   * Convert the resulting matrix into a DataFrame for better visualization.

4. **Output**

   * Print the vocabulary dictionary.
   * Display the BoW representation (raw counts).
   * Display the TF-IDF representation (weighted values).

---

## PROGRAM

```python
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import pandas as pd

# Sample corpus
corpus = [
    "Harry likes to play football.",
    "Mary enjoys playing tennis.",
    "Football and tennis are popular sports."
]

# Bag of Words (BoW)
count_vectorizer = CountVectorizer()
bow_matrix = count_vectorizer.fit_transform(corpus)
bow_df = pd.DataFrame(bow_matrix.toarray(), columns=count_vectorizer.get_feature_names_out())

print("=== Bag of Words (BoW) ===")
print("Vocabulary:", count_vectorizer.vocabulary_)
print("\nBoW Representation:\n", bow_df)

# TF-IDF Representation
tfidf_vectorizer = TfidfVectorizer()
tfidf_matrix = tfidf_vectorizer.fit_transform(corpus)
tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=tfidf_vectorizer.get_feature_names_out())

print("\n=== TF-IDF ===")
print("Vocabulary:", tfidf_vectorizer.vocabulary_)
print("\nTF-IDF Representation:\n", tfidf_df.round(2))
```

---

## OUTPUT

```
=== Bag of Words (BoW) ===
Vocabulary: {'harry': 4, 'likes': 5, 'to': 12, 'play': 7, 'football': 3, 'mary': 6, 'enjoys': 2, 'playing': 8, 'tennis': 11, 'and': 0, 'are': 1, 'popular': 9, 'sports': 10}

BoW Representation:
    and  are  enjoys  football  harry  likes  mary  play  playing  popular  \
0    0    0       0         1      1      1     0     1        0        0   
1    0    0       1         0      0      0     1     0        1        0   
2    1    1       0         1      0      0     0     0        0        1   

   sports  tennis  to  
0       0       0   1  
1       0       1   0  
2       1       1   0  

=== TF-IDF ===
Vocabulary: {'harry': 4, 'likes': 5, 'to': 12, 'play': 7, 'football': 3, 'mary': 6, 'enjoys': 2, 'playing': 8, 'tennis': 11, 'and': 0, 'are': 1, 'popular': 9, 'sports': 10}

TF-IDF Representation:
     and   are  enjoys  football  harry  likes  mary  play  playing  popular  \
0  0.00  0.00    0.00      0.36   0.47   0.47  0.00  0.47     0.00     0.00   
1  0.00  0.00    0.53      0.00   0.00   0.00  0.53  0.00     0.53     0.00   
2  0.44  0.44    0.00      0.33   0.00   0.00  0.00  0.00     0.00     0.44   

   sports  tennis    to  
0    0.00    0.00  0.47  
1    0.00    0.40  0.00  
2    0.44    0.33  0.00  
```

---

