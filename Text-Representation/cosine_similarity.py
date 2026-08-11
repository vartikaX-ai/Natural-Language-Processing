from sklearn.metrics.pairwise import cosine_similarity
from gensim.models import Word2Vec
from nltk.tokenize import word_tokenize

sentences = [
    "I love NLP",
    "I love machine learning",
    "NLP is interesting",
    "Machine learning is useful"
]
word = [word_tokenize(sentence.lower()) for sentence in sentences]
word2vec = Word2Vec(word,window=2,vector_size=100,min_count=1,negative=5,sg=1,epochs=10)
word2vec.build_vocab(word)
word2vec.train(word,total_examples=word2vec.corpus_count,epochs=word2vec.epochs)

vectorA = word2vec.wv.get_vector("machine").reshape(1,-1)
vectorB = word2vec.wv.get_vector("learning").reshape(1,-1)

cosineSimilarity = cosine_similarity(vectorA,vectorB)
print("Cosine Similarity: ")
print(cosineSimilarity)