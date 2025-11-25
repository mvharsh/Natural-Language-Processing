# Stemming & Lemmatization

## **AIM**

To demonstrate text normalization in NLP by applying **Stemming** and **Lemmatization** on a given sentence using Python’s NLTK library.

---

## **ALGORITHM**

1. Import required modules from NLTK:

   * `word_tokenize`, `PorterStemmer`, `WordNetLemmatizer`, `pos_tag`, `wordnet`.
2. Download necessary NLTK data: `'punkt'`, `'wordnet'`, `'averaged_perceptron_tagger'`.
3. Define the input text.
4. Tokenize the sentence using `word_tokenize()`.
5. Apply stemming using `PorterStemmer`:

   * Stem each token.
6. Apply lemmatization (default mode, assuming noun).
7. Perform POS tagging using `pos_tag()`.
8. Map POS tags from Treebank format to WordNet format using a helper function.
9. Apply lemmatization with POS tags.
10. Print results for each step.

---

## **PROGRAM**

```python
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.corpus import wordnet
from nltk import pos_tag
import nltk

text = "The striped bats are hanging on their feet for best."

# Word Tokenization
words = word_tokenize(text)

# Stemming
stemmer = PorterStemmer()
stemmed_words = [stemmer.stem(word) for word in words]
print("Stemmed Words:\n", stemmed_words, "\n")

# Lemmatization
lemmatizer = WordNetLemmatizer()
lemmatized_words = [lemmatizer.lemmatize(word) for word in words]
print("Lemmatized Words (default - noun):\n", lemmatized_words, "\n")

# POS tag mapping for lemmatization
def get_wordnet_pos(treebank_tag):
    if treebank_tag.startswith('J'):
        return wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return wordnet.VERB
    elif treebank_tag.startswith('N'):
        return wordnet.NOUN
    elif treebank_tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN

# Lemmatization with POS tagging
pos_tags = pos_tag(words)
lemmatized_pos = [lemmatizer.lemmatize(word, get_wordnet_pos(pos)) for word, pos in pos_tags]
print("Lemmatized Words (with POS):\n", lemmatized_pos)
```

---

## **OUTPUT**

```
Stemmed Words:
['the', 'stripe', 'bat', 'are', 'hang', 'on', 'their', 'feet', 'for', 'best', '.']

Lemmatized Words (default - noun):
['The', 'striped', 'bat', 'are', 'hanging', 'on', 'their', 'foot', 'for', 'best', '.']

Lemmatized Words (with POS):
['The', 'striped', 'bat', 'be', 'hang', 'on', 'their', 'foot', 'for', 'best', '.']
```

---

