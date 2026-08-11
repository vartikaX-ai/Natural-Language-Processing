# NLP Text Representation

## About

Text representation is the process of converting human-readable text into numerical representations that can be processed by machine learning and deep learning models.

This folder contains implementations of common text representation techniques, including sparse representations and dense word embeddings.

---

## Techniques Covered

### 1. One-Hot Encoding

One-Hot Encoding represents each unique word using a binary vector.

Each word is assigned a vector containing:

* `1` at the position corresponding to that word
* `0` at all other positions

**Example:**

```text
Vocabulary: ["cat", "dog", "tiger"]

cat   → [1, 0, 0]
dog   → [0, 1, 0]
tiger → [0, 0, 1]
```

**Limitation:** The vectors are sparse and do not capture semantic relationships between words.

---

### 2. Bag of Words (BoW)

Bag of Words represents a document based on the occurrence of words, without considering their order.

**Example:**

```text
"I love NLP"
"I love ML"
```

The model creates a vocabulary and represents each document using word counts.

**Characteristics:**

* Simple and easy to implement
* Produces sparse vectors
* Ignores word order
* Does not capture semantic meaning

---

### 3. N-Grams

N-Grams represent consecutive sequences of words.

Common types include:

* **Unigram** → one word
* **Bigram** → two consecutive words
* **Trigram** → three consecutive words

**Example:**

```text
"I love natural language processing"
```

Bigrams:

```text
"I love"
"love natural"
"natural language"
"language processing"
```

N-Grams can capture some local word-order information that Bag of Words cannot.

---

### 4. TF-IDF

**TF-IDF (Term Frequency–Inverse Document Frequency)** represents words based on how important they are within a document and across a collection of documents.

It consists of:

* **TF (Term Frequency)** → how frequently a term appears in a document
* **IDF (Inverse Document Frequency)** → reduces the importance of terms that appear in many documents

A common form of the formula is:

```text
TF-IDF = TF × IDF
```

TF-IDF generally produces sparse numerical representations.

---

## Word Embeddings

Word embeddings represent words as **dense numerical vectors** in a continuous vector space.

Unlike One-Hot Encoding, embeddings can capture relationships between words based on their usage and context.

Examples:

* Word2Vec
* GloVe
* FastText

---

### 5. Word2Vec

Word2Vec learns dense word representations based on relationships between words and their surrounding context.

It has two main architectures:

* **CBOW (Continuous Bag of Words)** → predicts a target word from surrounding context words.
* **Skip-gram** → predicts surrounding context words from a target word.

Word2Vec produces **static embeddings**, meaning a word generally has the same vector regardless of the sentence in which it appears.

---

### 6. GloVe

**GloVe (Global Vectors for Word Representation)** learns word embeddings using **global word co-occurrence statistics** from a corpus.

Words that frequently occur in similar contexts tend to obtain similar vector representations.

GloVe also produces **static embeddings**.

---

### 7. FastText

FastText extends the idea of word embeddings by representing words using **character n-grams (subwords)**.

For example, a word can be represented using smaller character sequences.

This allows FastText to:

* Capture subword information
* Handle rare words better
* Generate representations for some out-of-vocabulary words

FastText is particularly useful for morphologically rich languages and words with meaningful subword structure.

---

## Sparse vs Dense Representations

| Representation   | Type   | Semantic Meaning |
| ---------------- | ------ | ---------------- |
| One-Hot Encoding | Sparse | ❌                |
| Bag of Words     | Sparse | ❌                |
| N-Grams          | Sparse | Limited          |
| TF-IDF           | Sparse | Limited          |
| Word2Vec         | Dense  | ✅                |
| GloVe            | Dense  | ✅                |
| FastText         | Dense  | ✅                |

---

## Comparison

| Technique | Representation | Considers Word Order | Captures Semantics |
| --------- | -------------- | -------------------- | ------------------ |
| One-Hot   | Sparse         | ❌                    | ❌                  |
| BoW       | Sparse         | ❌                    | ❌                  |
| N-Grams   | Sparse         | ✅ Local order        | Limited            |
| TF-IDF    | Sparse         | ❌                    | Limited            |
| Word2Vec  | Dense          | Context-based        | ✅                  |
| GloVe     | Dense          | Co-occurrence-based  | ✅                  |
| FastText  | Dense          | Subword-based        | ✅                  |

---

## Cosine Similarity

Cosine similarity can be used to measure the similarity between numerical vectors, including word embeddings.

A higher cosine similarity generally indicates that two vectors point in more similar directions.

It can be used for tasks such as:

* Finding similar words
* Comparing text representations
* Measuring semantic similarity

---

## Implementations

The practical implementations in this folder include:

* One-Hot Encoding
* Bag of Words
* N-Grams
* TF-IDF
* Word2Vec
* FastText
* Cosine Similarity
* BoW vs TF-IDF comparison

GloVe is covered at the conceptual level.
