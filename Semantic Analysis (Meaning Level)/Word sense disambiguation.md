# WORD SENSE DISAMBIGUATION

## AIM

To implement Word Sense Disambiguation (WSD) using the Lesk algorithm in NLTK to identify the correct sense of an ambiguous word from a given sentence.

---

## ALGORITHM

1. Import the WordNet corpus and the Lesk algorithm from NLTK.
2. Define sentences containing an ambiguous word (e.g., *bat*).
3. Apply the Lesk algorithm to determine the most appropriate sense of the word in each sentence.
4. Retrieve the word sense and its definition from WordNet.
5. Display the disambiguated sense and its meaning for each occurrence of the word.

---

## PROGRAM

```python
from nltk.corpus import wordnet as wn
from nltk.wsd import lesk

sentence2 = "The bat flied inside the room"
word2 = "bat"
sentence = "He buy a new bat to play cricket"
word = "bat"

# Apply Lesk algorithm
sense = lesk(sentence.split(), word)
print("Word:", word)
print("Sense:", sense)
print("Definition:", sense.definition())

sense = lesk(sentence2.split(), word)
print("Word:", word2)
print("Sense:", sense)
print("Definition:", sense.definition())
```

---

## OUTPUT

```
Word: bat
Sense: Synset('bat.v.04')
Definition: use a bat
Word: bat
Sense: Synset('cricket_bat.n.01')
Definition: the club used in playing cricket
```

---
