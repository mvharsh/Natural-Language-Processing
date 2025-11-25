# CHUNKING

## AIM

To implement a simple Noun Phrase (NP) chunking algorithm in NLP using NLTK’s RegexpParser, which identifies and extracts noun phrases from a given sentence based on part-of-speech tags.

---

## ALGORITHM

1. Input a sentence.
2. Tokenize the sentence into words.
3. Apply POS tagging to assign part-of-speech tags.
4. Define a grammar rule for noun phrases (e.g., `<DT>?<JJ>*<NN.*>`).
5. Parse the POS-tagged sentence using the grammar.
6. Extract and print NP subtrees from the parsed tree.

---

## PROGRAM

```python
import nltk

sentence = "The quick brown fox jumps over the lazy dog"

# Tokenize and POS tagging
tokens = nltk.word_tokenize(sentence)
pos_tags = nltk.pos_tag(tokens)

# Define a simple grammar for NP (Noun Phrase) chunking
grammar = r"""
  NP: {<DT>?<JJ>*<NN.*>}   # Determiner + Adjectives + Noun
"""

# Create a chunk parser
chunk_parser = nltk.RegexpParser(grammar)

# Parse sentence
tree = chunk_parser.parse(pos_tags)
print(tree)

# Extract and print NP chunks (no GUI)
for subtree in tree.subtrees(filter=lambda t: t.label() == 'NP'):
    chunk = " ".join(word for word, tag in subtree.leaves())
    print("Noun Phrase Chunk:", chunk)
```

---

## OUTPUT

```
(S
  (NP The/DT quick/JJ brown/NN)
  (NP fox/NN)
  jumps/VBZ
  over/IN
  (NP the/DT lazy/JJ dog/NN))

Noun Phrase Chunk: The quick brown
Noun Phrase Chunk: fox
Noun Phrase Chunk: the lazy dog
```

---

