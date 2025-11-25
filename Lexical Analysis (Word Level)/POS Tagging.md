# POS TAGGING

## AIM

To perform Part-of-Speech (POS) tagging on a given text using NLTK and SpaCy, identifying the grammatical category (noun, verb, adjective, etc.) of each word.

---

## ALGORITHM

### Using NLTK

1. Import necessary modules from **nltk**.
2. Download required resources: `'punkt'` for tokenization and `'averaged_perceptron_tagger'` for POS tagging.
3. Define the input text.
4. Tokenize the text using **word_tokenize()**.
5. Use **nltk.pos_tag()** to tag each token with its POS.
6. Print the results.

### Using SpaCy

1. Import the **spacy** library.
2. Load the pre-trained English language model: **en_core_web_sm**.
3. Define the input text.
4. Use the model to process the text and create a **doc** object.
5. Loop through each token in *doc* and print the POS tag using **token.pos_**.

---

## PROGRAM

### Using NLTK

```python
import nltk
from nltk.tokenize import word_tokenize

text = "Natural Language Processing enables computers to understand human language."

# Tokenize the text
tokens = word_tokenize(text)

# Perform POS tagging
pos_tags = nltk.pos_tag(tokens)

print("POS Tagged Output:")
for word, pos_tag in pos_tags:
    print(f"{word}: {pos_tag}")
```

### OUTPUT

```
POS Tagged Output:
Natural: JJ
Language: NNP
Processing: NNP
enables: VBZ
computers: NNS
to: TO
understand: VB
human: JJ
language: NN
.: .
```

---

### Using SpaCy

```python
#importing libraries 
import spacy

# Load the English language model
nlp = spacy.load("en_core_web_sm")

# Sample text
text = "Natural Language Processing enables computers to understand human language."

# Process the text with SpaCy
doc = nlp(text)

print("Original Text: ", text)
print("PoS Tagging Result:")
for token in doc:
    print(f"{token.text}: {token.pos_}")
```

### OUTPUT

```
Original Text:  Natural Language Processing enables computers to understand human language.
PoS Tagging Result:
Natural: PROPN
Language: PROPN
Processing: NOUN
enables: VERB
computers: NOUN
to: PART
understand: VERB
human: ADJ
language: NOUN
.: PUNCT
```

---


