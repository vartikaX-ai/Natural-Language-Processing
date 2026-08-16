import numpy as np
from keras.models import Model
from keras.layers import Dense,SimpleRNN,Input

input = np.array([[[1],[2],[3]],[[4],[5],[6]],[[7],[8],[9]],[[10],[11],[12]],[[13],[14],[15]],[[16],[17],[18]]])/18.0
target = np.array([[[3],[2],[1]],[[6],[5],[4]],[[9],[8],[7]],[[12],[11],[10]],[[15],[14],[13]],[[18],[17],[16]]])/18.0

encoder_input = Input(shape=(3,1))
encoder_output,encoder_state = SimpleRNN(64,return_state=True)(encoder_input)

decoder_input = Input(shape=(3,1))
decoder_sequence,decoder_state = SimpleRNN(64,return_sequences=True,return_state=True)(decoder_input,initial_state=encoder_state)

output = Dense(1)(decoder_sequence)

model = Model(inputs=[encoder_input,decoder_input],outputs=output)

model.compile(optimizer="adam",loss="mse",metrics=["mae"])

decoder_data = np.zeros_like(input)
model.fit([input,decoder_data],target,epochs=100)

test_encoder = np.array([[[10],[11],[12]]])/18.0
test_decoder = np.array([[[0],[0],[0]]])
pred = model.predict([test_encoder,test_decoder])*18.0
print(pred)