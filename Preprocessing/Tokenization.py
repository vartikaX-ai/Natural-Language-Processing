from nltk.tokenize import word_tokenize,sent_tokenize

document = "I am learning Natural Language Processing with Python."

sentences = sent_tokenize(document)

words = word_tokenize(document)

print("Sentence Tokenization: ")
print(sentences)

print("Word Tokenization: ")
print(words)