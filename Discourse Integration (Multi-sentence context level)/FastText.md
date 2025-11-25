# FASTTEXT EMBEDDINGS

## AIM

To implement **FastText embeddings** using Gensim in Python to generate word vectors, capture semantic similarity, and handle **unseen words** through subword information.

---

## ALGORITHM

1. **Prepare Training Data**

   * Define a list of tokenized sentences related to celestial bodies (e.g., sun, moon, stars, earth, planets).

2. **Train FastText Model**

   * Use `FastText()` from Gensim with parameters:

     * `vector_size=50` → embedding dimensions
     * `window=3` → context window size
     * `min_count=1` → include all words
     * `sg=1` → Skip-gram model
     * `epochs=30` → training iterations

3. **Retrieve Word Vectors**

   * Access vector representations using `model.wv["word"]`.
   * Print the first few dimensions for verification.

4. **Find Similar Words**

   * Use `most_similar()` to identify semantically closest words (e.g., earth, stars).

5. **Handle Unseen Words**

   * Test FastText’s ability to generate vectors for unseen words (e.g., sunlight) using subword embeddings.

---

## PROGRAM

```python
from gensim.models import FastText

# Step 1: Training sentences
sentences = [
    ["the", "sun", "rises", "in", "the", "east"],
    ["the", "moon", "shines", "at", "night"],
    ["stars", "twinkle", "in", "the", "sky"],
    ["earth", "revolves", "around", "the", "sun"],
    ["planets", "are", "part", "of", "the", "solar", "system"]
]

# Step 2: Train FastText model
model = FastText(sentences, vector_size=50, window=3, min_count=1, sg=1, epochs=30)

# Step 3: Retrieve vector for a word
print("FastText vector for 'sun':", model.wv["sun"][:5], "...")  # first 5 dims

# Step 4: Find most similar words
print("Most similar to 'earth':", model.wv.most_similar("earth"))
print("Most similar to 'stars':", model.wv.most_similar("stars"))

# Step 5: Test unseen word (subword embeddings)
print("FastText vector for unseen word 'sunlight':", model.wv["sunlight"][:5], "...")
```

---

## OUTPUT

```
FastText vector for 'sun': [ 0.00770814  0.00627485  0.00177704  0.00087953 -0.01016692] ...

Most similar to 'earth': [('part', 0.34136), ('around', 0.18017), ('east', 0.16023), ('in', 0.15181), ('stars', 0.12415), ...]

Most similar to 'stars': [('earth', 0.12415), ('twinkle', 0.12199), ('planets', 0.06517), ('moon', 0.06087), ('are', 0.05440), ...]

FastText vector for unseen word 'sunlight': [ 0.00313646  0.00071503 -0.0004184 -0.00013085 -0.00004715] ...
```

---

