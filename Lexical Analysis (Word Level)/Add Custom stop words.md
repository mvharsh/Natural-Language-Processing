# ADD CUSTOM STOP WORDS

## AIM

To filter out custom-defined stop words from a given text using basic Python operations without relying on built-in NLTK stop word lists.

---

## ALGORITHM

1. Define the input text.
2. Define a custom set of stop words that may be unnecessary for a specific analysis.
3. Split the text into words using `.split()`.
4. For each word:

   * Strip punctuation like commas and periods using `.strip(".,")`.
   * Convert to lowercase for uniform comparison.
   * Check if the cleaned word is **not** in the `custom_stopwords` set.
5. Add non-stop words to a filtered list.
6. Print:

   * The count of filtered words.
   * The final filtered word list.

---

## PROGRAM

```python
text = """Artificial Intelligence and Machine Learning are transforming the way we live and work."""

custom_stopwords = {"and", "the", "we", "are", "to", "on", "as", "of", "in", "it", "is", "an", "also", "they", "such", "these", "from", "while", "becoming", "that", "our"}

words = text.split()
filtered = [word.strip(".,") for word in words if word.lower().strip(".,") not in custom_stopwords]

print("Words after removal: ", len(filtered))
print(filtered)
```

---

## OUTPUT

```
Words after removal:  8
['Artificial', 'Intelligence', 'Machine', 'Learning', 'transforming', 'way', 'live', 'work']
```
