# NLP Repository

This repository contains a collection of **Natural Language Processing (NLP)** implementations covering preprocessing, lexical & syntactic analysis, semantic analysis, discourse integration, word embeddings, and sentiment analysis. It is intended for educational purposes, demonstrations, and experimentation with various NLP techniques.

---

## **Contents of the Repository**

### **1. Text Preprocessing & Cleaning (Pre-Linguistic / Preprocessing)**

* **Remove Punctuations & Numbers** – Clean raw text data.
* **Tokenization** – Split text into words or sentences.
* **Spelling Correction** – Correct common spelling mistakes.

### **2. Lexical Analysis (Word Level)**

* **Stemming & Lemmatization** – Reduce words to their root/base form.
* **Stop Word Identification & Removal** – Remove common, non-informative words.
* **Add Custom Stop Words** – Extend the stop word list as per dataset needs.
* **POS Tagging** – Assign Part-of-Speech tags to words.
* **POS Tagging - Ambiguous** – Handle words with multiple possible POS tags.

### **3. Syntactic Analysis (Grammar / Sentence Level)**

* **Syntax Parsing (CFG)** – Analyze sentence structure using Context-Free Grammars.
* **Brill Tagger** – Rule-based POS tagging approach.
* **HMM** – Hidden Markov Model for sequence labeling.
* **Viterbi Algorithm** – Dynamic programming algorithm for optimal sequence tagging.
* **Chunking** – Extract phrases (like NP, VP) from tagged text.

### **4. Semantic Analysis (Meaning Level)**

* **Word Sense Disambiguation** – Resolve ambiguity in word meanings.
* **Named Entity Recognition (NER)** – Identify entities like names, locations, dates.
* **Semantic Role Labeling (SRL)** – Assign semantic roles to sentence constituents.

### **5. Discourse Integration (Multi-sentence Context Level)**

* **Discourse Markers & Hobbs's Algorithm** – Analyze discourse structure and anaphora resolution.
* **Text Representation** – Convert text into numerical features.

  * **Bag-of-Words (BOW) with TF-IDF**
  * **Word2Vec**
  * **GloVe**
  * **FastText**
* **N-Gram Model** – Model word sequences for prediction or analysis.

### **6. Sentiment Analysis (Subjectivity & Application Level)**

* **Hate Speech Detection** – Detect offensive or hateful content in text.
* **Movie Review Analysis** – Classify movie reviews as positive/negative.
* **Aspect-Based Sentiment Analysis** – Determine sentiment for specific aspects in text.

### **7. Miniproject**

* **Efficient Fine-Tuning Techniques for Tamil Text Summarization** 




## **Requirements**

* Python 3.7+
* Libraries: `nltk`, `spacy`, `gensim`, `sklearn`, `pandas`, `numpy`, `textblob`, `transformers`, `peft`, `torch`, `datasets`, `rouge_score`, `xgboost`, `bitsandbytes`

---


