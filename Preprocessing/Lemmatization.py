from nltk.stem import WordNetLemmatizer

words = ["playing", "played", "studies", "studying", "better", "children"]

wordnet = WordNetLemmatizer()

for word in words:
    print(word,"->",wordnet.lemmatize(word,pos="n"))