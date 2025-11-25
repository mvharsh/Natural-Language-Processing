# SEMANTIC ROLE LABELING (SRL)

## AIM

To implement a basic form of Semantic Role Labeling (SRL) using dependency parsing in SpaCy, where syntactic dependencies are mapped to semantic roles such as Agent, Theme, Recipient, and Time.

---

## ALGORITHM

1. Import the SpaCy library and load the pre-trained English model (`en_core_web_sm`).
2. Define an input sentence containing a verb and its arguments (e.g., “Rani sold the book to Raj yesterday.”).
3. Parse the sentence using the SpaCy NLP pipeline.
4. Create a mapping dictionary from dependency labels (e.g., `nsubj`, `dobj`, `pobj`) to semantic roles (e.g., ARG0-Agent, ARG1-Theme, ARG2-Recipient).
5. For each verb in the sentence, identify its children (subjects, objects, modifiers).
6. Map these syntactic dependencies to semantic roles and print the result.
7. Handle prepositional objects (like “to Raj”) explicitly by checking for prepositions and their objects.

---

## PROGRAM

```python
import spacy

# Load model
nlp = spacy.load("en_core_web_sm")

sentence = "Rani sold the book to Raj yesterday."
doc = nlp(sentence)

# Dependency → Semantic Role mapping
dep_to_arg = {
    "nsubj": "ARG0 (Agent)",
    "dobj": "ARG1 (Theme)",
    "pobj": "ARG2 (Recipient/Beneficiary)",
    "iobj": "ARG2 (Indirect Object)",
    "npadvmod": "ARGM-TMP (Time)",
    "advmod": "ARGM-MNR (Manner)",
    "prep": "ARG2 (Prepositional Object)"
}

print("Semantic Roles (approx via dependency parse):")
for token in doc:
    if token.pos_ == "VERB":
        print(f"\n[V: {token.text}]")
        for child in token.children:
            role = dep_to_arg.get(child.dep_, child.dep_)
            # Handle preposition "to Raj" properly
            if child.dep_ == "prep":
                for pobj in child.children:
                    if pobj.dep_ == "pobj":
                        print(f"[{dep_to_arg['pobj']}: {child.text} {pobj.text}]")
            else:
                print(f"[{role}: {child.text}]")
```

---

## OUTPUT

```
Semantic Roles (approx via dependency parse):

[V: sold]
[ARG0 (Agent): Rani]
[ARG1 (Theme): book]
[ARG2 (Recipient/Beneficiary): to Raj]
[ARGM-TMP (Time): yesterday]
[punct: .]
```

---
