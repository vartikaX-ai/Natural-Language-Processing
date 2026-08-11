from gensim.models import FastText

sentences = [
    ["i", "love", "playing", "football"],
    ["i", "am", "playing", "cricket"],
    ["players", "are", "playing", "well"],
    ["football", "players", "are", "strong"]
]

fasttext = FastText(sentences,vector_size=100,window=2,min_count=1,epochs=10)
fasttext.build_vocab(sentences)
fasttext.train(sentences,total_examples=fasttext.corpus_count,epochs=fasttext.epochs)

print("Embedding of playing: ",fasttext.wv["playing"])
print("Playfully exist in corpus: ","playfully" in sentences)
print("Embedding of playfully: ",fasttext.wv["playfully"])