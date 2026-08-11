# NLP Preprocessing

## About

Text preprocessing is the process of cleaning and transforming raw text into a format that can be effectively used by NLP models and algorithms.

This folder contains implementations of common NLP preprocessing techniques.

---

## Techniques Covered

### 1. Tokenization

Tokenization splits text into smaller units called **tokens**, such as words or sentences.

**Example:**

```text
"I love NLP"
        ↓
["I", "love", "NLP"]
```

Common types:

* Word Tokenization
* Sentence Tokenization

---

### 2. Lowercasing

Converts text to lowercase so that words with different capitalization are treated consistently.

```text
"I Love NLP"
      ↓
"i love nlp"
```

---

### 3. Stopword Removal

Stopwords are common words that often carry limited information for certain NLP tasks.

Examples:

```text
the, is, am, are, a, an, and, in
```

**Example:**

```text
"I am learning NLP"
        ↓
["learning", "NLP"]
```

Stopword removal should be applied based on the task because some stopwords can carry important meaning.

---

### 4. Punctuation Removal

Removes punctuation marks when they are not useful for the specific NLP task.

```text
"Hello, world!"
       ↓
"Hello world"
```

---

### 5. Stemming

Stemming reduces words to a common root form by removing prefixes or suffixes.

**Example:**

```text
playing  → play
played   → play
studies  → studi
```

Stemming is generally faster but may produce words that are not valid dictionary words.

---

### 6. Lemmatization

Lemmatization reduces a word to its **lemma**, or dictionary base form, using linguistic information.

**Example:**

```text
playing  → play
studies  → study
children → child
```

Lemmatization generally produces more meaningful results than stemming but can be more computationally expensive.

---

## Stemming vs Lemmatization

| Stemming                  | Lemmatization                  |
| ------------------------- | ------------------------------ |
| Uses word reduction rules | Uses linguistic analysis       |
| Faster                    | Usually slower                 |
| May produce invalid words | Produces meaningful base forms |
| Less accurate             | Generally more accurate        |

---

## Typical Preprocessing Pipeline

A basic NLP preprocessing pipeline can be:

```text
Raw Text
   ↓
Lowercasing
   ↓
Tokenization
   ↓
Punctuation Removal
   ↓
Stopword Removal
   ↓
Stemming / Lemmatization
   ↓
Clean Text
```

The exact preprocessing steps depend on the **NLP task, dataset, and model**. Not every preprocessing technique is required for every problem.

---

## Implementations

The practical implementations in this folder include:

* Tokenization
* Stopword Removal
* Stemming
* Lemmatization
* Stemming vs Lemmatization
* Complete Text Preprocessing Pipeline
