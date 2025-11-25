# DISCOURSE MARKERS & HOBBS-STYLE PRONOUN RESOLUTION

## AIM

To implement a program using SpaCy for detecting discourse markers and performing a simplified form of pronoun resolution (Hobbs-style approximation) in a given text.

---

## ALGORITHM

1. Import the SpaCy library and load the pre-trained English model (`en_core_web_sm`).
2. Define a set of discourse markers (e.g., however, because, although, but).
3. Create a function to find discourse markers:

   * Iterate over tokens in the parsed text.
   * Check if the token matches any predefined discourse marker.
   * Store and return all matched markers.
4. Create a function to resolve pronouns (simplified Hobbs-style):

   * Iterate over tokens in the text.
   * If a token is a pronoun (`PRON`) and functions as subject/object, capture it.
   * Find the head verb governing the pronoun (if any).
   * Store the pronoun with its associated head verb.
5. Define an input text containing discourse markers and pronouns.
6. Process the text with the SpaCy pipeline.
7. Extract discourse markers and pronoun references using the defined functions.
8. Print the extracted results.

---

## PROGRAM

```python
import spacy

# Load model
nlp = spacy.load("en_core_web_sm")

DISCOURSE_MARKERS = {"however", "therefore", "because", "but", "although", "though", "yet", "so"}

def find_discourse_markers(doc):
    markers = []
    for token in doc:
        if token.text.lower() in DISCOURSE_MARKERS:
            markers.append(token.text)
    return markers

def resolve_pronouns(doc):
    pronoun_refs = []
    for token in doc:
        if token.pos_ == "PRON" and token.dep_ in {"nsubj", "dobj", "nsubjpass"}:
            head_verb = token.head.text if token.head.pos_ == "VERB" else None
            pronoun_refs.append({
                "Pronoun": token.text,
                "Head Verb": head_verb
            })
    return pronoun_refs

# Input text
text = ("Harry studied hard for the exam. He passed with good marks. "
        "Because of this, he got a scholarship. Although he was nervous, he gave a great speech.")

doc = nlp(text)

markers = find_discourse_markers(doc)
pronoun_refs = resolve_pronouns(doc)

print("Discourse Markers Found")
for m in markers:
    print(m)

print("\nPronoun References")
for ref in pronoun_refs:
    print(f"Pronoun: {ref['Pronoun']}, Head Verb: {ref['Head Verb']}")
```

---

## OUTPUT

```
Discourse Markers Found
Because
Although

Pronoun References
Pronoun: He, Head Verb: passed
Pronoun: he, Head Verb: got
Pronoun: he, Head Verb: None
Pronoun: he, Head Verb: gave
```

---

