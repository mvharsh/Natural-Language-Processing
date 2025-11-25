# Spelling Correction

## **AIM**

To detect and correct spelling mistakes in a sentence using **TextBlob** and **pyspellchecker**.

---

## **ALGORITHM**

### **Using TextBlob**

1. Import `TextBlob`.
2. Define a misspelled sentence.
3. Use `TextBlob(text).correct()` to generate the corrected output.
4. Print original and corrected texts.

### **Using SpellChecker**

1. Import `SpellChecker`.
2. Define the same input text.
3. Split the text into words.
4. For each word, use `spell.correction()` to get the best guess.
5. Join the corrected words into a final sentence.
6. Use `spell.unknown()` to list all misspelled words.

---

## **PROGRAM**

### **Using TextBlob**

```python
from textblob import TextBlob

text = "I havv goood speling in my txt mesage."
corrected_text = TextBlob(text).correct()

print("Original:", text)
print("Corrected:", corrected_text)
```

### **Output:**

```
Original: I havv goood speling in my txt mesage.
Corrected: I have good spelling in my txt message.
```

---

### **Using SpellChecker**

```python
from spellchecker import SpellChecker

spell = SpellChecker()

text = "I havv goood speling in my txt mesage"
words = text.split()

corrected_words = []
for word in words:
    corrected_words.append(spell.correction(word))

corrected_text = ' '.join(corrected_words)

print("Original:", text)
print("Corrected:", corrected_text)

misspelled = spell.unknown(words)
print("Misspelled words:", misspelled)
```

### **Output:**

```
Original: I havv goood speling in my txt mesage
Corrected: I have good spelling in my text message
Misspelled words: {'mesage', 'havv', 'goood', 'txt', 'speling'}
```

---
