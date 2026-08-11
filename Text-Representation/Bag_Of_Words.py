from sklearn.feature_extraction.text import CountVectorizer

sentences = [
    "I love NLP",
    "I love machine learning",
    "NLP is interesting"
]

cv = CountVectorizer()
bow = cv.fit_transform(sentences)

print("Vocabulary: ",cv.get_feature_names_out())
print("Document Term Matrix: ",bow.toarray())