# Remove Punctuations and Numbers

## **AIM**

To remove punctuation marks and numbers from a given text using regular expressions in Python.

## **ALGORITHM**

1. Input the original text containing words, punctuation, and numbers.
2. Use the `re.sub()` function with a regular expression pattern:

   * `[^A-Za-z\s]` matches any character not in A–Z, a–z, or whitespace (`\s`).
   * This effectively removes digits (0–9) and punctuation.
3. Store the cleaned output text.
4. Print both the original and cleaned texts.

## **PROGRAM**

```python
import re

text = "Natural Language Processing (NLP) in 2025 is amazing! It combines AI, ML, & data science."

# Remove punctuations and numbers
cleaned_text = re.sub(r'[^A-Za-z\s]', '', text)

print("Original Text:")
print(text)
print("\nCleaned Text:")
print(cleaned_text)
```

## **OUTPUT**

```
Original Text:
Natural Language Processing (NLP) in 2025 is amazing! It combines AI, ML, & data science.

Cleaned Text:
Natural Language Processing NLP in  is amazing It combines AI ML  data science
```

