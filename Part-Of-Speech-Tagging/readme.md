# NLP POS Tagging

## About

POS (Part-of-Speech) Tagging is the process of assigning a grammatical category to each word in a sentence.

This folder contains an implementation of POS tagging using **spaCy** to identify the grammatical role of each token in a sentence.

---

## POS Tags Covered

Common POS tags include:

* **NOUN** → Noun
* **VERB** → Verb
* **ADJ** → Adjective
* **ADV** → Adverb
* **PRON** → Pronoun
* **DET** → Determiner
* **ADP** → Adposition
* **AUX** → Auxiliary Verb
* **PROPN** → Proper Noun
* **NUM** → Numeral
* **PART** → Particle
* **INTJ** → Interjection

---

## POS Tagging

POS Tagging identifies the grammatical role of each word based on its context.

**Example:**

```text
"I love machine learning"
```

The sentence can be represented as:

```text
I          → PRON
love       → VERB
machine    → NOUN
learning   → NOUN
```

POS tagging helps NLP systems understand the grammatical structure of text.

---

## Tokenization vs POS Tagging

Tokenization breaks text into individual tokens.

```text
"I love NLP"
↓
["I", "love", "NLP"]
```

POS Tagging assigns a grammatical category to each token.

```text
I     → PRON
love  → VERB
NLP   → NOUN
```

Therefore:

**Tokenization → identifies the tokens**
**POS Tagging → identifies their grammatical roles**

---

## Context in POS Tagging

The grammatical role of a word can change depending on the context.

**Example:**

```text
"I can book a flight."
"I read a book."
```

Here, `book` has different grammatical roles:

```text
"I can book a flight."
          ↓
         VERB

"I read a book."
          ↓
         NOUN
```

This shows why context is important in POS tagging.

---

## Practical Implementation

### POS Tagging using spaCy

Implemented a POS tagging system using the pretrained **spaCy `en_core_web_sm`** model.

The system:

1. Loads the pretrained spaCy model.
2. Processes the input sentence.
3. Tokenizes the sentence.
4. Assigns a POS tag to each token.
5. Displays the POS tag and its description.

### Input

```text
"Elon Musk founded SpaceX in California in 2002."
```

### Workflow

```text
Input Sentence
      ↓
spaCy NLP Pipeline
      ↓
Tokenization + POS Tagging
      ↓
Token + POS Tag + Description
```

---

## Model Used

* spaCy `en_core_web_sm`

---

## Evaluation

POS tagging is evaluated by comparing the predicted POS tags with the correct grammatical tags.

For this implementation, the pretrained spaCy model performs the POS tagging directly.

---

## Libraries Used

* spaCy

---

## POS Tag Information

The implementation uses:

```python
token.pos_
```

to obtain the coarse-grained POS tag.

The POS tag description can be obtained using:

```python
spacy.explain(token.pos_)
```

---

## POS Tagging vs Named Entity Recognition

| Task        | Purpose                     | Example          |
| ----------- | ---------------------------- | ----------------- |
| POS Tagging | Identifies grammatical role | `founded → VERB` |
| NER         | Identifies named entities    | `SpaceX → ORG`    |

POS tagging focuses on **grammatical categories**, while NER focuses on **real-world entities**.

---

## Applications

POS Tagging is used in:

* Information Extraction
* Named Entity Recognition
* Question Answering
* Machine Translation
* Text Parsing
* Lemmatization
* Grammar Analysis
* Text Analysis

---

## Implementations

The practical implementation in this folder includes:

* POS Tagging using spaCy
* Token-level POS identification
* POS tag descriptions
* Context-based grammatical classification
