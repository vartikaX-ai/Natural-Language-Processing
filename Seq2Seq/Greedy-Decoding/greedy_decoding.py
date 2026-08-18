import numpy as np
from keras.layers import LSTM,Embedding,Dense,Input
from keras.models import Model
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

encod_seq = [
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
tokenizer.fit_on_texts(encod_seq+decoded_in+decoded_target)

encod_seq = tokenizer.texts_to_sequences(encod_seq)
decoded_in = tokenizer.texts_to_sequences(decoded_in)
decoded_target = tokenizer.texts_to_sequences(decoded_target)

max_len = 5

encod_seq = pad_sequences(encod_seq,maxlen=3,padding="post")
decoded_in = pad_sequences(decoded_in,maxlen=max_len,padding="post")
decoded_target = pad_sequences(decoded_target,maxlen=max_len,padding="post")

vocab_size = len(tokenizer.word_index)+1

encoded_input = Input(shape=(3,))
embedd_encod = Embedding(vocab_size,32)(encoded_input)
encoded_output,encoded_h,encoded_c = LSTM(64,return_sequences=True,return_state=True)(embedd_encod)

decoded_input = Input(shape=(5,))
embedd_decod = Embedding(vocab_size,32)(decoded_input)
decoded_sequences,decoded_h,decoded_c = LSTM(64,return_sequences=True,return_state=True)(embedd_decod,initial_state=[encoded_h,encoded_c])

output = Dense(vocab_size,activation="softmax")(decoded_sequences)

model = Model(inputs=[encoded_input,decoded_input],outputs=output)

model.compile(optimizer="adam",loss="sparse_categorical_crossentropy",metrics=["accuracy"])

model.fit([encod_seq,decoded_in],decoded_target,epochs=100)

index_to_word = {
    value : key
    for key,value in tokenizer.word_index.items()
}

def greedy_decode(input_text,max_decode_len=5):
    input_text = tokenizer.texts_to_sequences(input_text)
    input_text = pad_sequences(input_text,maxlen=3,padding="post")

    sos_token = tokenizer.word_index["sos"]
    eos_token = tokenizer.word_index["eos"]

    decode_data = np.array([[sos_token]+[0]*(max_len-1)])

    predicted_word = []

    for i in range(max_decode_len):
        probabilities = model.predict([input_text,decode_data])
        next_token = np.argmax(probabilities[0,i,:])
        if next_token == eos_token:
            break
        word = index_to_word.get(next_token,"")
        if word:
            predicted_word.append(word)
        if i+1<max_len:
            decode_data[0][i+1] = next_token
    return " ".join(predicted_word)

test_data = ["cat sat mat"]
result = greedy_decode(test_data)
print(result)