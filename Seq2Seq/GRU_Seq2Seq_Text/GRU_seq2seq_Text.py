import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense,GRU,Input,Embedding
from keras.models import Model

encoded_seq = [
    "one two three",
    "cat mat sat",
    "red green blue",
    "big small tall"
]

decoded_in = [
    "<SOS> three two one",
    "<SOS> sat mat cat",
    "<SOS> blue green red",
    "<SOS> tall small big"
]

decoded_target = [
    "three two one <EOS>",
    "sat mat cat <EOS>",
    "blue green red <EOS>",
    "tall small big <EOS>"
]

tokenizer = Tokenizer()
tokenizer.fit_on_texts(encoded_seq+decoded_in+decoded_target)

encoded_seq = tokenizer.texts_to_sequences(encoded_seq)
decoded_in = tokenizer.texts_to_sequences(decoded_in)
decoded_target = tokenizer.texts_to_sequences(decoded_target)

encoded_seq = pad_sequences(encoded_seq,padding="post")
decoded_in = pad_sequences(decoded_in,padding="post")
decoded_target = pad_sequences(decoded_target,padding="post")

vocab_size = len(tokenizer.word_index)+1

encoded_input = Input(shape=(3,))
embedd_encod = Embedding(vocab_size,32)(encoded_input)
encoded_output,encoded_state = GRU(64,return_state=True)(embedd_encod)

decoded_input = Input(shape=(4,))
embedd_decod = Embedding(vocab_size,32)(decoded_input)
decoded_sequences,decoded_state = GRU(64,return_sequences=True,return_state=True)(embedd_decod,initial_state=encoded_state)

output = Dense(vocab_size,activation="softmax")(decoded_sequences)

model = Model(inputs=[encoded_input,decoded_input],outputs=output)

model.compile(optimizer="adam",loss="sparse_categorical_crossentropy",metrics=["accuracy"])

model.fit([encoded_seq,decoded_in],decoded_target,epochs=100)

test_data = ["red green blue"]
test_data = tokenizer.texts_to_sequences(test_data)
test_data = pad_sequences(test_data,maxlen=3,padding="post")

sos_token = tokenizer.word_index["sos"]
decoded_data = np.array([[sos_token,0,0,0]])

pred = model.predict([test_data,decoded_data])
prediction_ids = np.argmax(pred[0],axis=-1)

index_to_words = {
    value : key
    for key,value in tokenizer.word_index.items()
}

prediction_words = [index_to_words.get(i,"") for i in prediction_ids]
print(prediction_words)