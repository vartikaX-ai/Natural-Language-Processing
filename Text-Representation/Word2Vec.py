from gensim.models import Word2Vec
from nltk.tokenize import word_tokenize

sentences = [
    "I love NLP",
    "I love machine learning",
    "NLP is interesting",
    "Machine learning is useful"
]

words = [word_tokenize(sentence.lower()) for sentence in sentences]

word2vec =  Word2Vec(vector_size=100,window=2,min_count=1,sg=1,negative=5,epochs=10)
word2vec.build_vocab(words)
word2vec.train(words,total_examples=word2vec.corpus_count,epochs=word2vec.epochs)

print(word2vec.wv.key_to_index)
print("Embedding of NLP: ",word2vec.wv["nlp"])
print("Embedding dimension",len(word2vec.wv["nlp"]))
print("Similar words of machine: ",word2vec.wv.most_similar("machine"))