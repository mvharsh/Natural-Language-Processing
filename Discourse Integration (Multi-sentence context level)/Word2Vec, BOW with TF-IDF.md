# WORD2VEC (WITH BOW / TF-IDF CONTEXT)

## AIM

To implement **Word2Vec embeddings** using the Gensim library in Python to represent words as dense vectors and compute semantic similarity between them.

---

## ALGORITHM

1. **Prepare the Corpus**

   * Define a list of tokenized sentences as the training data.

2. **Initialize and Train Word2Vec Model**

   * Use Gensim’s `Word2Vec()` with parameters:

     * `vector_size=50` → dimensionality of word embeddings.
     * `window=3` → context window size.
     * `min_count=1` → minimum frequency for words to be included.
     * `workers=4` → number of worker threads.
     * `sg=0` → CBOW (Continuous Bag of Words) model.
   * Train the model on the given sentences.

3. **Extract Word Embeddings**

   * Retrieve the vector representation of a word (e.g., `"football"`) using `model.wv["word"]`.
   * Print a portion of the vector to verify.

4. **Find Semantic Similarities**

   * Use `most_similar()` to find the top-N words most similar to a given word (e.g., `"sports"`).
   * Print the similar words with their similarity scores.

---

## PROGRAM

```python
from gensim.models import Word2Vec

# Tokenized corpus
sentences = [
    ["harry", "likes", "to", "play", "football"],
    ["tom", "enjoys", "playing", "tennis"],
    ["football", "and", "tennis", "are", "popular", "sports"],
    ["harry", "and", "tom", "went", "to", "the", "stadium"],
    ["sports", "bring", "people", "together"]
]

# Train Word2Vec model (CBOW by default, use sg=1 for Skip-gram)
model = Word2Vec(sentences, vector_size=50, window=3, min_count=1, workers=4, sg=0)

# Retrieve vector for a specific word
word_vector = model.wv["football"]
print("Word2Vec vector for 'football':\n", word_vector[:10], "...")  # print first 10 dims

# Find most similar words
similar_words = model.wv.most_similar("sports", topn=5)
print("\nMost similar to 'sports':")
for word, score in similar_words:
    print(f"{word}: {score:.2f}")
```

---

## OUTPUT

```
Word2Vec vector for 'football':
 [ 0.01563514 -0.01902037 -0.00041106  0.00693839 -0.00187794  0.01676354
  0.01802157  0.01307301 -0.00142324  0.01542081] ...

Most similar to 'sports':
bring: 0.23
together: 0.15
tom: 0.12
enjoys: 0.08
harry: 0.07
```

---
