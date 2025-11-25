# BRILL TAGGER

## AIM

To train a Brill POS tagger using the Penn Treebank corpus and evaluate both its accuracy and similarity to the ground truth using cosine similarity of tag distributions.

---

## ALGORITHM

1. **Load Dataset:** Import and split the Treebank tagged sentences into training and test sets.
2. **Train Base Tagger:** Train a `UnigramTagger` as the initial model.
3. **Train Brill Tagger:** Use Brill transformation rules (`fntbl37`) to improve tagging accuracy on top of the unigram model.
4. **Evaluate Accuracy:** Calculate the accuracy of the Brill tagger on the test set.
5. **Compute Cosine Similarity:**
   a) Flatten the actual and predicted tags.
   b) Encode tags as integers using `LabelEncoder`.
   c) Convert them into tag frequency vectors.
   d) Compute cosine similarity between true and predicted tag distributions.

---

## PROGRAM

```python
import nltk
from nltk.corpus import treebank
from nltk.tag import UnigramTagger, brill, brill_trainer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from sklearn.preprocessing import LabelEncoder
from collections import Counter

# Download necessary resources
nltk.download('treebank')

# Load and split data
tagged_sentences = treebank.tagged_sents()
train_data = tagged_sentences[:3000]
test_data = tagged_sentences[3000:]

# Train UnigramTagger
unigram_tagger = UnigramTagger(train_data)

# Define templates for BrillTagger
templates = brill.fntbl37()

# Train Brill Tagger
trainer = brill_trainer.BrillTaggerTrainer(initial_tagger=unigram_tagger, templates=templates, trace=0)
brill_tagger = trainer.train(train_data)

# Accuracy Evaluation
accuracy = brill_tagger.evaluate(test_data)
print("Brill Tagger Accuracy:", round(accuracy * 100, 2), "%")

# Cosine Similarity as Similarity Index
def flatten_tags(data):
    return [tag for sent in data for (word, tag) in sent]

true_tags = flatten_tags(test_data)
predicted_tags = flatten_tags([brill_tagger.tag([w for w, _ in sent]) for sent in test_data])

# Encode tags as integers
le = LabelEncoder()
all_tags = list(set(true_tags + predicted_tags))
le.fit(all_tags)
true_encoded = le.transform(true_tags)
pred_encoded = le.transform(predicted_tags)

# Convert to one-hot frequency vectors
def tag_freq_vector(tag_list):
    counts = Counter(tag_list)
    vector = [counts.get(tag, 0) for tag in range(len(all_tags))]
    return np.array(vector).reshape(1, -1)

true_vec = tag_freq_vector(true_encoded)
pred_vec = tag_freq_vector(pred_encoded)

similarity = cosine_similarity(true_vec, pred_vec)[0][0]
print("Cosine Similarity (Tag Distribution):", round(similarity, 4))
```

---

## OUTPUT

```
Brill Tagger Accuracy: 87.63 %
Cosine Similarity (Tag Distribution): 0.9195
```

---

