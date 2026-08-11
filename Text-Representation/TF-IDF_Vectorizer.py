from sklearn.feature_extraction.text import TfidfVectorizer

documents = [
    "I love machine learning",
    "I love deep learning",
    "Machine learning is interesting"
]

tv = TfidfVectorizer()
tfidf = tv.fit_transform(documents)

print("Vocabulary: ",tv.get_feature_names_out())
print("TF-IDF Matrix: ",tfidf.toarray())