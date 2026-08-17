import tensorflow as tf
import numpy as np
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from keras.models import Model
from keras.layers import Embedding,Dense,LSTM,Input
from keras.losses import SparseCategoricalCrossentropy

encoded_seq = [
    "one two three",
    "sat mat",
    "red green blue",
    "big small"
]

decoded_in = [
    "<SOS> three two one",
    "<SOS> mat sat",
    "<SOS> blue green red",
    "<SOS> small big"
]

decoded_target = [
    "three two one <EOS>",
    "mat sat <EOS>",
    "blue green red <EOS>",
    "small big <EOS>"
]

tokenizer = Tokenizer()
tokenizer.fit_on_texts(encoded_seq+decoded_in+decoded_target)
encoded_seq = tokenizer.texts_to_sequences(encoded_seq)
decoded_in = tokenizer.texts_to_sequences(decoded_in)
decoded_target = tokenizer.texts_to_sequences(decoded_target)

max_encoded_len = max(len(x) for x in encoded_seq)
max_decoded_len = max(len(x) for x in decoded_in)
encoded_seq = pad_sequences(encoded_seq,maxlen=max_encoded_len,padding="post")
decoded_in = pad_sequences(decoded_in,maxlen=max_decoded_len,padding="post")
decoded_target = pad_sequences(decoded_target,maxlen=max_decoded_len,padding="post")

vocab_size = len(tokenizer.word_index)+1

encoded_input = Input(shape=(max_encoded_len,))
embedd_encod = Embedding(vocab_size,32)(encoded_input)
encoded_output,encoded_h,encoded_c = LSTM(64,return_state = True)(embedd_encod)

decoded_input = Input(shape=(max_decoded_len,))
embedd_decod = Embedding(vocab_size,32)(decoded_input)
decoded_sequence,decoded_h,decoded_c = LSTM(64,return_sequences=True,return_state=True)(embedd_decod,initial_state=[encoded_h,encoded_c])

output = Dense(vocab_size,activation="softmax")(decoded_sequence)

model = Model(inputs=[encoded_input,decoded_input],outputs=output)

loss_function = SparseCategoricalCrossentropy(from_logits=False,reduction="none")

def masked_loss(y_true,y_pred):

    loss = loss_function(y_true,y_pred)
    mask = tf.cast(
        tf.not_equal(y_true,0),
        tf.float32
    )

    loss = loss*mask
    return tf.reduce_sum(loss)/tf.reduce_sum(mask)

model.compile(optimizer="adam",loss=masked_loss,metrics=["accuracy"])

model.fit([encoded_seq,decoded_in],decoded_target,epochs=100)

test_data = ["sat mat"]
test_data = tokenizer.texts_to_sequences(test_data)
test_data = pad_sequences(test_data,maxlen=max_encoded_len,padding="post")

sos_token = tokenizer.word_index["sos"]
decoded_data = np.array([[sos_token]+[0]*(max_decoded_len-1)])

pred = model.predict([test_data,decoded_data])
prediction_ids = np.argmax(pred[0],axis=-1)

index_to_word = {
    value : key
    for key,value in tokenizer.word_index.items()
}

prediction_words = [index_to_word.get(i,"") for i in prediction_ids]
print(prediction_words)