from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

stop_words = set(stopwords.words("English"))
sentence = "I am learning NLP and I am enjoying it."

words = word_tokenize(sentence)

word = [word for word in words if word not in stop_words]
print(word)