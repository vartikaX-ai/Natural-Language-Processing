import numpy as np
from keras.models import Model
from keras.layers import Dense,LSTM,Input

input = np.array([[[1],[2],[3]],[[4],[5],[6]],[[7],[8],[9]],[[10],[11],[12]],[[13],[14],[15]],[[16],[17],[18]]])/18.0
target = np.array([[[3],[2],[1]],[[6],[5],[4]],[[9],[8],[7]],[[12],[11],[10]],[[15],[14],[13]],[[18],[17],[16]]])/18.0

encoded_input = Input(shape=(3,1))
encoded_output,encoded_h,encoded_c = LSTM(64,return_state=True)(encoded_input)

decoded_input = Input(shape=(3,1))
decoded_sequence,decoded_h,decoded_c = LSTM(64,return_sequences=True,return_state=True)(decoded_input,initial_state=[encoded_h,encoded_c])

output = Dense(1)(decoded_sequence)

model = Model(inputs=[encoded_input,decoded_input],outputs=output)

model.compile(optimizer="adam",loss="mse",metrics=["mae"])

decoded_data = np.zeros_like(input)
model.fit([input,decoded_data],target,epochs=100)

test_encoded_data = np.array([[[11],[12],[13]]])/18.0
test_decoded_data = np.array([[[0],[0],[0]]])

pred = model.predict([test_encoded_data,test_decoded_data])*18.0
print(pred)