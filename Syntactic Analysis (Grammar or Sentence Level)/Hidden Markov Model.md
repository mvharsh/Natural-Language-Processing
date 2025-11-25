# HIDDEN MARKOV MODEL (HMM)

## AIM

To implement and evaluate a Hidden Markov Model (HMM) POS tagger using the Penn Treebank dataset and test its performance on a sample sentence.

---

## ALGORITHM

1. **Import Libraries & Download Data:** Import necessary modules from `nltk` and download the Treebank corpus.
2. **Prepare Dataset:** Load and split the tagged sentences into training and testing sets.
3. **Train HMM Tagger:** Use `HiddenMarkovModelTrainer` to train the HMM model in supervised mode on the training data.
4. **Evaluate Accuracy:** Evaluate the trained HMM tagger on the test set and print the accuracy.
5. **Tag Sentence:** Tokenize a test sentence and tag it using the trained HMM tagger.

---

## PROGRAM

```python
import nltk
from nltk.corpus import treebank
from nltk.tag.hmm import HiddenMarkovModelTrainer

# Download data
nltk.download('treebank')

# Load and split the tagged data
tagged_sentences = treebank.tagged_sents()
train_data = tagged_sentences[:3000]
test_data = tagged_sentences[3000:]

# Create and train the HMM tagger
trainer = HiddenMarkovModelTrainer()
hmm_tagger = trainer.train_supervised(train_data)

# Evaluate on the test set
accuracy = hmm_tagger.evaluate(test_data)
print("HMM Tagger Accuracy:", round(accuracy * 100, 2), "%")

# Try tagging a custom sentence
sentence = ["There", "is", "a", "cat", "on", "the", "table", "."]
tagged = hmm_tagger.tag(sentence)
print("Tagged Output:", tagged)
```

---

## OUTPUT

```
HMM Tagger Accuracy: 36.84 %

Tagged Output: [('There', 'EX'), ('is', 'VBZ'), ('a', 'DT'), ('cat', 'NNP'), ('on', 'NNP'), ('the', 'NNP'), ('table', 'NNP'), ('.', 'NNP')]
```

---
