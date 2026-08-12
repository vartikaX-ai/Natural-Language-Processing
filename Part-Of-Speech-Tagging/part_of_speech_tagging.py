import spacy

nlp = spacy.load("en_core_web_sm")

sentence = "Elon Musk founded SpaceX in California in 2002."

doc = nlp(sentence)

for token in doc:
    print("Token: ",token.text)
    print("POS tag: ",token.pos_)
    print("Description: ",spacy.explain(token.pos_))