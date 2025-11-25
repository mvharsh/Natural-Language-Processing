# GLOVE WORD EMBEDDINGS

## AIM

To implement and explore **GloVe word embeddings** in Python by loading pre-trained vectors, retrieving word representations, and finding semantically similar words using **cosine similarity**.

---

## ALGORITHM

1. **Download and Extract GloVe**

   * If `glove.6B.50d.txt` is not present, download and unzip the pre-trained embeddings.

2. **Load GloVe Embeddings**

   * Define a function `load_glove(filepath)` to read the file.
   * Store each word as a key and its 50-dimensional vector as a NumPy array in a dictionary.

3. **Cosine Similarity Function**

   * Define `cosine_similarity(vec1, vec2)` to compute similarity between two vectors.

4. **Most Similar Words Function**

   * Define `most_similar(word, embeddings, top_k=5)`.
   * Retrieve the vector of the given word.
   * Compute cosine similarity with all other words in the vocabulary.
   * Sort and return the top-k most similar words.

5. **Run Experiments**

   * Retrieve embeddings for words like `university`, `love`, `music`.
   * Print the first few dimensions of each vector.
   * Display the most similar words for each query word based on cosine similarity.

---

## PROGRAM

```python
import os
import numpy as np

# Step 1: Download GloVe if not already present
if not os.path.exists("glove.6B.50d.txt"):
    !wget http://nlp.stanford.edu/data/glove.6B.zip
    !unzip glove.6B.zip

# Step 2: Load GloVe embeddings
def load_glove(filepath):
    embeddings = {}
    with open(filepath, encoding="utf8") as f:
        for line in f:
            values = line.split()
            word = values[0]
            vector = np.asarray(values[1:], dtype="float32")
            embeddings[word] = vector
    return embeddings

# Step 3: Cosine similarity
def cosine_similarity(vec1, vec2):
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

# Step 4: Find most similar words
def most_similar(word, embeddings, top_k=5):
    if word not in embeddings:
        return f"'{word}' not found in vocabulary."
    word_vec = embeddings[word]
    similarities = {}
    for other, vec in embeddings.items():
        if other != word:
            similarities[other] = cosine_similarity(word_vec, vec)
    return sorted(similarities.items(), key=lambda x: x[1], reverse=True)[:top_k]

# ---- Run Experiment ----
glove_file = "glove.6B.50d.txt"
embeddings = load_glove(glove_file)

# Query words
query_words = ["university", "love", "music"]

for w in query_words:
    if w in embeddings:
        print(f"\nVector for '{w}':", embeddings[w][:5], "...")  
        print(f"Most similar to '{w}':", most_similar(w, embeddings))
    else:
        print(f"'{w}' not found in embeddings.")
```

---

## OUTPUT

```
Vector for 'university': [-1.1082   1.2916  -0.78751 -0.45955 -0.40788] ...
Most similar to 'university': [('college', 0.87446), ('harvard', 0.87106), ('yale', 0.85668), ('graduate', 0.85529), ('institute', 0.84836)]

Vector for 'love': [-0.13886  1.1401  -0.85212 -0.29212  0.75534] ...
Most similar to 'love': [('dream', 0.84296), ('life', 0.84034), ('dreams', 0.83986), ('loves', 0.83611), ('me', 0.83518)]

Vector for 'music': [-0.92448   0.59807  -0.995    -0.045298 -0.38836 ] ...
Most similar to 'music': [('musical', 0.88535), ('pop', 0.86821), ('dance', 0.85312), ('songs', 0.85255), ('recording', 0.83919)]
```

---
