# SYNTAX PARSING

## AIM

To perform syntactic parsing of a natural language sentence using Context-Free Grammar (CFG) and display all possible parse trees using NLTK’s ChartParser.

---

## ALGORITHM

1. **Import Libraries:** Import required modules from the `nltk` library, including `CFG` and `ChartParser`.
2. **Define Grammar:** Create a context-free grammar (CFG) using production rules to define sentence structures (e.g., `S -> NP VP`).
3. **Create Parser:** Instantiate a `ChartParser` using the defined CFG.
4. **Tokenize Sentence:** Tokenize the input sentence using `nltk.word_tokenize()`.
5. **Parse Tokens:** Apply the chart parser to the tokenized sentence to generate all valid parse trees.
6. **Display Output:** Pretty print all the generated parse trees to visualize syntactic structures.

---

## PROGRAM

```python
import nltk
from nltk import CFG
from nltk.parse.chart import ChartParser

# Download required resources
nltk.download('punkt_tab')

# Define the CFG (Context-Free Grammar)
grammar = CFG.fromstring("""
S -> NP VP
NP -> Det N | Det N PP | 'I'
VP -> V NP | VP PP
PP -> P NP

Det -> 'a' | 'the'
N -> 'man' | 'park' | 'dog' | 'telescope'
V -> 'saw' | 'walked'
P -> 'in' | 'with'
""")

# Create a parser using the CFG
parser = ChartParser(grammar)

# Tokenize and parse a sentence
sentence = "I saw a man with a telescope"
tokens = nltk.word_tokenize(sentence)

# Generate and display all parse trees
print(f"Parsing the sentence: {sentence}\n")
for tree in parser.parse(tokens):
    tree.pretty_print()
```

---

## OUTPUT

```
Parsing the sentence: I saw a man with a telescope

     S                                  
  ___|___________                         
 |               VP                      
 |        _______|________                
 |       VP               PP             
 |    ___|___         ____|___          
 |   |       NP      |        NP        
 |   |    ___|___    |     ___|______    
 NP  V  Det      N   P   Det         N  
 |   |   |       |   |    |          |   
 I  saw  a      man with  a      telescope

     S                              
  ___|_______                         
 |           VP                      
 |    _______|___                     
 |   |           NP                  
 |   |    _______|____                
 |   |   |   |        PP             
 |   |   |   |    ____|___            
 |   |   |   |   |        NP          
 |   |   |   |   |     ___|______    
 NP  V  Det  N   P   Det         N  
 |   |   |   |   |    |          |   
 I  saw  a  man with  a      telescope
```

---

