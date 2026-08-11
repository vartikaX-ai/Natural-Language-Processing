import string as st
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

text = "I am learning NLP! It is an amazing field, and I am really enjoying learning it."

#Convert Text into Lowercase
text = text.lower()
#Removing Punctuation
translator = text.maketrans("","",st.punctuation)
text = text.translate(translator)
#Tokenize
words = word_tokenize(text)
#Stopword Removal
stop_words = set(stopwords.words("English"))
words = [word for word in words if word not in stop_words]
#Lemmatization
wordnet = WordNetLemmatizer()

for word in words:
    print(word,"->",wordnet.lemmatize(word,pos="v"))