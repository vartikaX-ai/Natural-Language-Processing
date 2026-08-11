# Named Entity Recognition (NER)

# About this Repository

This repository contains my implementation of **Named Entity Recognition (NER)** using **spaCy** and a pretrained NLP model.

NER is an NLP task that identifies named entities in text and assigns them predefined entity categories.

## Topics Covered

- Named Entity Recognition (NER)
- Entity Extraction
- Entity Classification
- Pretrained NLP Models
- spaCy
- Entity Labels

## Practical Implementation

### Named Entity Recognition

Implemented an NER system using the pretrained **`en_core_web_sm`** model from spaCy.

The system extracts named entities from text and displays:

- Entity
- Entity Label
- Label Description

### Example Input
Elon Musk founded SpaceX in California in 2002.
The company later announced a $1 billion investment in Tesla.

Example Entities
Elon Musk   → PERSON
SpaceX      → ORG
California  → GPE
2002        → DATE
$1 billion  → MONEY
Tesla       → ORG

Libraries Used
spaCy
Model Used
en_core_web_sm

The pretrained spaCy model is used to recognize and classify named entities without training an NER model from scratch.
