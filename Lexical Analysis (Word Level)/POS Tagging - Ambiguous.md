# POS TAGGING - AMBIGUOUS

## AIM

To demonstrate Part-of-Speech (POS) tagging on ambiguous words in sentences using Python’s NLTK and SpaCy, and observe how context determines their grammatical roles.

---

## ALGORITHM

### **POS Tagging using NLTK**

1. Import `word_tokenize` and `pos_tag` from `nltk`.
2. Download required NLTK resources: `punkt`, `averaged_perceptron_tagger`.
3. Define multiple input sentences containing ambiguous words.
4. For each sentence:

   * Tokenize it into words.
   * Apply `nltk.pos_tag()` to assign POS tags.
   * Print each word with its POS tag.

### **POS Tagging using SpaCy**

1. Import `spacy` and load the `en_core_web_sm` model.
2. For each sentence:

   * Pass it to the SpaCy NLP pipeline.
   * Loop through tokens and extract:

     * `token.text` → word
     * `token.pos_` → universal POS tag (NOUN, VERB, etc.)
     * `token.tag_` → fine‑grained tag (NN, VBZ, MD, etc.)
   * Print results.

---

## PROGRAM

### **Using NLTK:**

```python
import nltk
from nltk.tokenize import word_tokenize

# Ambiguous sentence examples
sentences = [
    "I can swim.",
    "He drank a can of soda.",
    "They can the vegetables.",
    "The flies are annoying.",
    "Time flies quickly.",
    "She will book a ticket.",
    "This book is interesting."
]

for sentence in sentences:
    tokens = word_tokenize(sentence)
    tagged = nltk.pos_tag(tokens)
    print(f"Sentence: {sentence}")
    print("POS Tags:", tagged)
    print()
```

### **OUTPUT (NLTK):**

```
Sentence: I can swim.
POS Tags: [('I', 'PRP'), ('can', 'MD'), ('swim', 'VB'), ('.', '.')]

Sentence: He drank a can of soda.
POS Tags: [('He', 'PRP'), ('drank', 'VBD'), ('a', 'DT'), ('can', 'MD'), ('of', 'IN'), ('soda', 'NN'), ('.', '.')]

Sentence: They can the vegetables.
POS Tags: [('They', 'PRP'), ('can', 'MD'), ('the', 'DT'), ('vegetables', 'NNS'), ('.', '.')]

Sentence: The flies are annoying.
POS Tags: [('The', 'DT'), ('flies', 'NNS'), ('are', 'VBP'), ('annoying', 'VBG'), ('.', '.')]

Sentence: Time flies quickly.
POS Tags: [('Time', 'NNP'), ('flies', 'VBZ'), ('quickly', 'RB'), ('.', '.')]

Sentence: She will book a ticket.
POS Tags: [('She', 'PRP'), ('will', 'MD'), ('book', 'NN'), ('a', 'DT'), ('ticket', 'NN'), ('.', '.')]

Sentence: This book is interesting.
POS Tags: [('This', 'DT'), ('book', 'NN'), ('is', 'VBZ'), ('interesting', 'JJ'), ('.', '.')]
```

---

### **Using SpaCy:**

```python
import spacy

# Load English language model
nlp = spacy.load("en_core_web_sm")

sentences = [
    "I can swim.",
    "He drank a can of soda.",
    "They can the vegetables.",
    "The flies are annoying.",
    "Time flies quickly.",
    "She will book a ticket.",
    "This book is interesting."
]

for sentence in sentences:
    doc = nlp(sentence)
    print(f"Sentence: {sentence}")
    for token in doc:
        print(f"{token.text:12} -> {token.pos_:<10} ({token.tag_})")
    print()
```

### **OUTPUT (SpaCy):**

```
Sentence: I can swim.
I            -> PRON       (PRP)
can          -> AUX        (MD)
swim         -> VERB       (VB)
.            -> PUNCT      (.)

Sentence: He drank a can of soda.
He           -> PRON       (PRP)
drank        -> VERB       (VBD)
a            -> DET        (DT)
can          -> NOUN       (NN)
of           -> ADP        (IN)
soda         -> NOUN       (NN)
.            -> PUNCT      (.)

Sentence: They can the vegetables.
They         -> PRON       (PRP)
can          -> AUX        (MD)
the          -> DET        (DT)
vegetables   -> NOUN       (NNS)
.            -> PUNCT      (.)

Sentence: The flies are annoying.
The          -> DET        (DT)
flies        -> NOUN       (NNS)
are          -> AUX        (VBP)
annoying     -> ADJ        (JJ)
.            -> PUNCT      (.)

Sentence: Time flies quickly.
Time         -> NOUN       (NN)
flies        -> VERB       (VBZ)
quickly      -> ADV        (RB)
.            -> PUNCT      (.)

Sentence: She will book a ticket.
She          -> PRON       (PRP)
will         -> AUX        (MD)
book         -> VERB       (VB)
a            -> DET        (DT)
ticket       -> NOUN       (NN)
.            -> PUNCT      (.)

Sentence: This book is interesting.
This         -> DET        (DT)
book         -> NOUN       (NN)
is           -> AUX        (VBZ)
interesting  -> ADJ        (JJ)
.            -> PUNCT      (.)
```

---

