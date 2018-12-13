from keras.layers import Dense
from keras.layers import LSTM

words = []

inputs = Input(shape=(len(words),))
x = Bidirectional(LSTM(128), input_shape=(maxlen, len(words)))(inputs)
x = Dropout(0.5)(x)
x = Bidirectional(LSTM(128), input_shape=(maxlen, len(words)))(inputs)
x = Dropout(0.5)(x)
predictions = Dense(len(words), activation='softmax')(x)
model = Model(inputs=inputs, outputs=predictions)
model.compile(optimizer='adam',
              loss='categorical_crossentropy')

model.fit(x, y, batch_size=128,
          epochs=60,
          callbacks=[print_callback])

