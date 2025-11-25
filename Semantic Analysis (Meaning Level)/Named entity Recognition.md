# NAMED ENTITY RECOGNITION (NER)

## AIM

To implement Named Entity Recognition (NER) using SpaCy to identify and classify entities such as names, dates, organizations, and locations from a given text.

---

## ALGORITHM

1. Import the SpaCy library and load the pre-trained English language model (`en_core_web_sm`).
2. Define the input text containing information with various entities (e.g., person names, dates, organizations, awards).
3. Pass the text to the SpaCy NLP pipeline to process it.
4. Extract named entities from the processed document using `doc.ents`.
5. Display each entity along with its predicted label (e.g., PERSON, DATE, ORG, GPE, NORP, etc.).

---

## PROGRAM

```python
import spacy

# Load SpaCy English model
nlp = spacy.load("en_core_web_sm")

text = """A.P.J. Abdul Kalam (1931–2015) 2025, June 5th, was an Indian aerospace engineer, statesman, 
and the 11th President of India (2002–2007). Known as the Missile Man of India for his work in missile 
and space technology at ISRO and DRDO, he was pivotal in India's nuclear tests and the development 
of its first satellite launch vehicle. A recipient of the Bharat Ratna, he became the People's President 
for his humility, dedication to education, and his inspiring books like Wings of Fire."""

# Process text
doc = nlp(text)

# Extract and print named entities
for ent in doc.ents:
    print(ent.text, ent.label_)
```

---

## OUTPUT

```
A.P.J. Abdul Kalam PERSON
1931–2015) 2025 DATE
June 5th DATE
Indian NORP
11th ORDINAL
India GPE
2002–2007 CARDINAL
ISRO ORG
DRDO ORG
India GPE
first ORDINAL
Wings of Fire ORG
```

---

