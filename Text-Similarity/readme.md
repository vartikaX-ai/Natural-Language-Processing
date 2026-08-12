# NLP Text Similarity

## About

Text similarity is the process of measuring how similar two pieces of text are based on their lexical or semantic meaning.

This folder contains implementations of text similarity techniques, including lexical similarity, semantic similarity, sentence embeddings, and cosine similarity.

## Techniques Covered

### 1. Lexical Similarity

Lexical similarity measures how similar two pieces of text are based mainly on the words they contain.

Common techniques include:
- Bag of Words
- TF-IDF
- N-Grams

**Example:**
- "I love machine learning"
- "I love deep learning"

These sentences have lexical similarity because they share some of the same words.

**Limitation:** Lexical methods may fail to recognize similar meanings when different words are used.

### 2. Semantic Similarity

Semantic similarity measures how similar two pieces of text are based on their meaning, rather than only their exact words.

**Example:**
- "I love machine learning"
- "I enjoy studying machine learning"

Although the wording is different, both sentences express a similar meaning.

Semantic similarity can be measured using sentence embeddings.

### 3. Sentence Embeddings

Sentence embeddings represent an entire sentence as a dense numerical vector.

A sentence is passed through a pretrained model and converted into a vector representation.

**Example:**
"I love machine learning"
↓
Sentence Transformer
↓
[0.21, -0.43, 0.67, ..., 0.15]

Sentence embeddings can capture semantic information and are useful for comparing sentences and documents.

### 4. Cosine Similarity

Cosine similarity measures the similarity between two numerical vectors by calculating the cosine of the angle between them.

A common formula is:
Cosine Similarity = (A · B) / (||A|| × ||B||)

A higher cosine similarity generally indicates that two vectors point in more similar directions.

Cosine similarity can be used for:
- Finding similar words
- Comparing sentence embeddings
- Comparing documents
- Measuring semantic similarity

## Sentence Transformers

Sentence Transformers are pretrained Transformer-based models designed to generate meaningful embeddings for sentences and other text.

They convert sentences into dense vector representations that can be compared using similarity measures such as cosine similarity.

## Sparse vs Dense Representations

| Representation       | Type   | Semantic Meaning |
|-----------------------|--------|-------------------|
| Bag of Words          | Sparse | ❌                |
| N-Grams               | Sparse | Limited           |
| TF-IDF                | Sparse | Limited           |
| Sentence Embeddings   | Dense  | ✅                |

## Comparison

| Technique            | Representation | Considers Word Meaning | Typical Use              |
|-----------------------|-----------------|--------------------------|----------------------------|
| BoW                   | Sparse          | ❌                       | Basic text matching       |
| N-Grams               | Sparse          | Limited                  | Phrase matching           |
| TF-IDF                | Sparse          | Limited                  | Document similarity       |
| Word Embeddings       | Dense           | ✅                       | Word-level similarity     |
| Sentence Embeddings   | Dense           | ✅                       | Sentence-level similarity |

## Practical Implementation

### Semantic Text Similarity

Implemented a semantic text similarity system using a pretrained Sentence Transformer model.

The system:
1. Takes two sentences as input.
2. Converts both sentences into sentence embeddings.
3. Calculates cosine similarity between the embeddings.
4. Produces a similarity score.

**Example Input:**

Sentence 1:
> I love machine learning and artificial intelligence.

Sentence 2:
> I enjoy studying machine learning and AI.

**Workflow:**
Sentence 1 ──→ Sentence Embedding ──┐
                                    ↓
                             Cosine Similarity
                                    ↓
                             Similarity Score
                                    ↑
Sentence 2 ──→ Sentence Embedding ──┘

### Model Used

- Pretrained Sentence Transformer

### Evaluation

The similarity between the two sentences is measured using Cosine Similarity.

The resulting similarity score represents how similar the two sentence embeddings are.

## Libraries Used

- Sentence Transformers
- Scikit-learn

## Implementations

The practical implementations in this folder include:

- Lexical Similarity
- Semantic Similarity
- Sentence Embeddings
- Cosine Similarity
- Sentence Transformer based Text Similarity
