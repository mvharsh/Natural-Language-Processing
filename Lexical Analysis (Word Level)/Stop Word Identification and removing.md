# Stop Word Identification and Removal

## **AIM**

To identify and remove stop words from a given text using Python’s NLTK library, leaving behind only meaningful words.

---

## **ALGORITHM**

1. Import required modules from **nltk**:

   * `word_tokenize`
   * `stopwords`
2. Download necessary NLTK data:

   * `'punkt'` for tokenization
   * `'stopwords'` for English stop words list
3. Define the input text.
4. Tokenize the text into words using `word_tokenize()`.
5. Convert each word to lowercase and:

   * Check if it is **not** in the list of English stop words.
   * Check if the word is **alphanumeric** (to exclude punctuation).
6. Filter and store the cleaned list of words.
7. Print:

   * The list of stop words
   * Total number of words after stop word removal
   * The final filtered list of words

---

## **PROGRAM**

```python
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

nltk.download('punkt')
nltk.download('stopwords')

text = """Artificial Intelligence and Machine Learning are transforming the way we live and work."""

words = word_tokenize(text)
filtered = [word for word in words if word.lower() not in stopwords.words('english') and word.isalnum()]

print('Stop words: ')
print(stopwords.words('english'))

print("Words after removal: ", len(filtered))
print(filtered)
```

---

## **OUTPUT**

```
Stop words:
['a', 'about', 'above', 'after', 'again', 'against', 'ain', 'all', 'am', 'an', 'and', 'any', 'are', ...]

Words after removal:  8
['Artificial', 'Intelligence', 'Machine', 'Learning', 'transforming', 'way', 'live', 'work']
```

---
