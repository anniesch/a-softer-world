import csv
import glob
import os

import cv2
import numpy as np
from tqdm import tqdm
from nltk import word_tokenize
from keras.layers import Dense
from keras.layers import LSTM
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input

NUM_COMICS = 1248
NUM_PANELS = 3
MAX_SENTENCE_LEN = 60

panel_path = os.path.join('data', 'panels')
hand_path = os.path.join('data', 'hand_transcriptions')

full_texts = []
all_panels = []
for index in tqdm(range(1, NUM_COMICS), desc='raw data'):
    panel_images = [os.path.join(panel_path, 'panel_{:04d}_{}.jpg'.format(
        index, panel_index)) for panel_index in range(NUM_PANELS)]
    text_csv = os.path.join(hand_path, 'text_{:04d}.csv'.format(index))
    try:
        with open(text_csv) as f:
            raw_texts = list(csv.reader(f))[0]
            new_texts = [['<s>'] + word_tokenize(text.lower()) + ['</s>'] 
                         for text in raw_texts]
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
    for panel_image in panel_images:
        all_panels.append(image.load_img(panel_image, target_size=(224,224)))
    full_texts.extend(new_texts)

words = sorted(list(set([word for text in full_texts for word in text])))
word_to_index = {word: index for index, word in enumerate(words)}
index_to_word = {word: index for index, word in enumerate(words)}


context_texts = []
next_words = []
panels = []
for full_text, panel in tqdm(zip(full_texts, all_panels), desc='preprocess',
                             total=len(full_texts)):
    for context_size in range(1, len(full_text)):
        context_texts.append(full_text[:context_size])
        next_words.append(full_text[context_size])
        panels.append(panel)

x_text = np.zeros((len(context_texts), MAX_SENTENCE_LEN, len(words)), 
                  dtype=np.bool)
x_image_raw = np.zeros((len(panels), 3, 224, 224))
y = np.zeros((len(context_texts), len(words)), dtype=np.bool)

for index, data in tqdm(enumerate(zip(context_texts, next_words, panel)),
                        desc='vectorize', total=len(next_words)):
    context, next_word, panel = data
    for word_index, word in enumerate(context):
        x_text[index, word_index, word_to_index[word]] = 1
    y[index, word_to_index[next_word]] = 1
    x_image[index, :, :, :] = image.img_to_array(panel)

x_image = preprocess_input(x_image_raw)
import pdb; pdb.set_trace()

image_input = Input(shape=(3, 224, 224))
text_input = Input(shape=(MAX_SENTENCE_LEN, len(words)))
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

