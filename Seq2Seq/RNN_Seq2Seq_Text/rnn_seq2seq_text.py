import numpy as np
from keras.models import Model
from keras.layers import Dense,Embedding,SimpleRNN,Input
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer

enc_seq = [
    "one two three",
    "cat sat mat",
    "red blue green",
    "big small tall"
]

decoded_in = [
    "<SOS> three two one",
    "<SOS> mat sat cat",
    "<SOS> green blue red",
    "<SOS> tall small big"
]

decoded_target = [
    "three two one <EOS>",
    "mat sat cat <EOS>",
    "green blue red <EOS>",
    "tall small big <EOS>"
]

tokenizer = Tokenizer()
tokenizer.fit_on_texts(enc_seq+decoded_in+decoded_target)

enc_seq = tokenizer.texts_to_sequences(enc_seq)
decoded_in = tokenizer.texts_to_sequences(decoded_in)
decoded_target = tokenizer.texts_to_sequences(decoded_target)

encoded_seq = pad_sequences(enc_seq,padding="post")
decodedIn_seq = pad_sequences(decoded_in,padding="post")
decodedTar_seq = pad_sequences(decoded_target,padding="post")

vocab_size = len(tokenizer.word_index)+1

encoded_input = Input(shape=(3,))
embedd_encod = Embedding(vocab_size,32)(encoded_input)
encoded_output,encoded_state = SimpleRNN(64,return_state=True)(embedd_encod)

decoded_input = Input(shape=(4,))
embedd_decod = Embedding(vocab_size,32)(decoded_input)
decoded_sequence,decoded_state = SimpleRNN(64,return_sequences=True,return_state=True)(embedd_decod,initial_state=encoded_state)

output = Dense(vocab_size,activation="softmax")(decoded_sequence)

model = Model(inputs=[encoded_input,decoded_input],outputs=output)

model.compile(optimizer="adam",loss="sparse_categorical_crossentropy",metrics=["accuracy"])

model.fit([encoded_seq,decodedIn_seq],decodedTar_seq,epochs=100)

test_data = ["one two three"]
test_seq = tokenizer.texts_to_sequences(test_data)
test_seq = pad_sequences(test_seq,padding="post",maxlen=3)

sos_token = tokenizer.word_index["sos"]

decoded_text = np.array([[sos_token,0,0,0]])

pred = model.predict([test_seq,decoded_text])

predicted_ids = np.argmax(pred[0],axis=-1)

index_to_word = {
    value : key
    for key,value in tokenizer.word_index.items()
}

predicted_word = [index_to_word.get(i,"") for i in predicted_ids]
print(predicted_word)