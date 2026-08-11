from nltk.stem import PorterStemmer

words = ["playing", "played", "plays", "studies", "studying", "easily", "fairly"]

stemmer = PorterStemmer()

for word in words:
    print(word,"->",stemmer.stem(word))