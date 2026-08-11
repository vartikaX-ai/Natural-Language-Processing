from nltk.stem import PorterStemmer,WordNetLemmatizer

words = ["studies", "playing", "better", "fairly"]

stemmer = PorterStemmer()
wordnet = WordNetLemmatizer()

print("Porter Stemmer: ")
for word in words:
    print(word,":",stemmer.stem(word))

print("Word Net Lemmatization: ")
for word1 in words:
    print(word1,":",wordnet.lemmatize(word1))