from sklearn.feature_extraction.text import CountVectorizer

text = ["I love natural language processing"]

for n,gram in zip([(1,1),(2,2),(3,3)],["Unigram","Bigram","Trigram"]):
    cv = CountVectorizer(ngram_range=n)
    cv.fit_transform(text)
    print(gram,"->",cv.get_feature_names_out())