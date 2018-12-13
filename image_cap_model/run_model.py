import csv
import glob
import os
from nltk import word_tokenize
from keras.layers import Dense
from keras.layers import LSTM

NUM_COMICS = 1248
NUM_PANELS = 3

panel_path = os.path.join('data', 'panels')
hand_path = os.path.join('data', 'hand_transcriptions')

texts = []
panel_paths = []

for index in range(1, NUM_COMICS):
    panel_images = [os.path.join(panel_path, 'panel_{:04d}_{}'.format(
        index, panel_index)) for panel_index in range(NUM_PANELS)]
    text_csv = os.path.join(hand_path, 'text_{:04d}.csv'.format(index))
    try:
        with open(text_csv) as f:
            raw_texts = list(csv.reader(f))[0]
            new_texts = [word_tokenize(text.lower()) for text in raw_texts]
    except OSError:
        continue 

    # this is a mild hack: the annotations don't include which text boxes
    # go to which panel, so we assume one goes to each (usually the case)
    # and assign the remainder of the text boxes to the last panel (okay
    # assumption, since most panels in a given comic look virtually the same)
    if len(panel_images) > len(new_texts):
        panel_images = panel_images[:len(new_texts)]
    while len(panel_images) < len(new_texts):
        panel_images.append(panel_images[-1])
    panel_paths.extend(panel_images)
    texts.extend(new_texts)

words = sorted(list(set([word for text in texts for word in text])))



image_input = Input(shape=...)
text_input = Input(shape=(len(words),))
vgg = keras.applications.vgg16.VGG16(include_top=False, weights='imagenet',
                                     input_tensor=image_input)
vgg_dense = Dense(128)(vgg)
lstm = Bidirectional(LSTM(128), input_shape=(maxlen, len(words)),
                     initial_state=vgg_dense)(text_input)
predictions = Dense(len(words), activation='softmax')(lstm)
model = Model(inputs=(text_input, image_input), outputs=predictions)
model.compile(optimizer='adam',
              loss='categorical_crossentropy')

model.fit(x, y, batch_size=128,
          epochs=60,
          callbacks=[print_callback])

