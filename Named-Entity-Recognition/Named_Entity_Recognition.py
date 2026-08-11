import spacy

nlp = spacy.load("en_core_web_sm")

text = "Elon Musk founded SpaceX in California in 2002. The company later announced a $1 billion investment in Tesla."

doc = nlp(text)

for ent in doc.ents:
    print("Entity: ",ent.text)
    print("Label: ",ent.label_)
    print("Description: ",spacy.explain(ent.label_))
    print()