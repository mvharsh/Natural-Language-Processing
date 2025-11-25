# N-GRAM MODEL

## AIM

To implement **N-gram modeling** using NLTK in Python by generating **bigrams** and **trigrams** from a given sentence.

---

## ALGORITHM

1. **Import Libraries**

   * Import `ngrams` and `word_tokenize` from NLTK.

2. **Define Sentence**

   * Use the input sentence: `"Harry and Tom went to the stadium"`.

3. **Tokenization**

   * Tokenize the sentence into lowercase words using `word_tokenize()`.

4. **Generate N-grams**

   * **Bigrams (N=2):** Use `ngrams(tokens, 2)`.
   * **Trigrams (N=3):** Use `ngrams(tokens, 3)`.

5. **Convert to List**

   * Convert the generated n-grams into lists for easy display.

6. **Print Output**

   * Display the original sentence, bigrams, and trigrams.

---

## PROGRAM

```python
from nltk import ngrams, word_tokenize

sentence = "Harry and Tom went to the stadium"

# Tokenize sentence
tokens = word_tokenize(sentence.lower())

# Generate bigrams (N=2)
bigrams = list(ngrams(tokens, 2))

# Generate trigrams (N=3)
trigrams = list(ngrams(tokens, 3))

print("Sentence:", sentence)
print("\nBigrams:", bigrams)
print("Trigrams:", trigrams)
```

---

## OUTPUT

```
Sentence: Harry and Tom went to the stadium

Bigrams: [('harry', 'and'), ('and', 'tom'), ('tom', 'went'), ('went', 'to'), ('to', 'the'), ('the', 'stadium')]
Trigrams: [('harry', 'and', 'tom'), ('and', 'tom', 'went'), ('tom', 'went', 'to'), ('went', 'to', 'the'), ('to', 'the', 'stadium')]
```

---

