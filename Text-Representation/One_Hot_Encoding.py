sentence = "I love NLP"

vocalubary = list(set(sentence.split(" ")))
for w,word in enumerate(vocalubary):
    vector = [0]*len(vocalubary)
    vector[w] = 1
    print(word,"->",vector)