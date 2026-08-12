from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

sentence1 = input("Enter a sentence1: ")
sentence2 = input("Enter a sentence2: ")

sentences = [
    sentence1,
    sentence2
]

st = SentenceTransformer("all-MiniLM-L6-v2")

embeddings = st.encode(sentences)

cosineSimilarity = cosine_similarity([embeddings[0]],[embeddings[1]])
print("Similarity score: ",cosineSimilarity[0][0])