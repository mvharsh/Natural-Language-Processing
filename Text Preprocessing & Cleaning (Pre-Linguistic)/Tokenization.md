# Tokenization

## **AIM**

To demonstrate various tokenization techniques used in NLP such as sentence, word, character, subword, and n-gram tokenization.

## **ALGORITHM**

1. Import necessary libraries: `nltk`, `transformers`, `sklearn`, etc.
2. Provide a sample text paragraph.
3. Apply different tokenization methods:

   * **Sentence Tokenization** using `sent_tokenize`
   * **Word Tokenization** using `word_tokenize`
   * **Character Tokenization** using `list()`
   * **Subword Tokenization** using a BERT tokenizer (`AutoTokenizer`)
   * **N-gram Tokenization** using `ngrams()` for bigrams and trigrams
4. Print the results for each method.

## **PROGRAM**

```python
from nltk.tokenize import sent_tokenize, word_tokenize
from sklearn.feature_extraction.text import CountVectorizer
from transformers import AutoTokenizer
from nltk.util import ngrams

text = """Natural language processing (NLP) is a field of computer science, artificial intelligence and computational linguistics concerned with the interactions between computers and human (natural) languages, and, in particular, concerned with programming computers to fruitfully process large natural language corpora. Challenges in natural language processing frequently involve natural language understanding, natural language generation (frequently from formal, machine-readable logical forms), connecting language and machine perception, managing human-computer dialog systems, or some combination thereof."""

# Sentence Tokenization
sentences = sent_tokenize(text)
print("Sentence Tokenization:\n", sentences, "\n")

# Word Tokenization
words = word_tokenize(text)
print("Word Tokenization:\n", words, "\n")

# Character Tokenization
characters = list(text)
print("Character Tokenization:\n", characters[:100], "...(truncated)\n")

# Subword Tokenization using a pretrained tokenizer
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
subwords = tokenizer.tokenize(text)
print("Subword Tokenization:\n", subwords, "\n")

# N-gram Tokenization (bigrams and trigrams)
bigrams = list(ngrams(words, 2))
trigrams = list(ngrams(words, 3))
print("Bigrams:\n", bigrams[:10], "...(truncated)\n")
print("Trigrams:\n", trigrams[:10], "...(truncated)\n")
```

## **OUTPUT (Truncated for readability)**

### **Sentence Tokenization:**

```
['Natural language processing (NLP) is a field of computer science ...',
 'Challenges in natural language processing frequently involve ...']
```

### **Word Tokenization:**

```
['Natural', 'language', 'processing', '(', 'NLP', ')', 'is', ...]
```

### **Character Tokenization:**

```
['N', 'a', 't', 'u', 'r', 'a', 'l', ' ', 'l', 'a', ...]
```

### **Subword Tokenization:**

```
['natural', 'language', 'processing', '(', 'nl', '##p', ')', ...]
```

### **Bigrams:**

```
[('Natural', 'language'), ('language', 'processing'), ...]
```

### **Trigrams:**

```
[('Natural', 'language', 'processing'), ('language', 'processing', '('), ...]
```

