import csv
import glob
import os

import numpy as np
from tqdm import tqdm
from nltk import word_tokenize
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Input 
from keras.layers import Flatten 
from keras.layers import Bidirectional 
from keras.layers import RepeatVector
from keras.layers import TimeDistributed 
from keras.layers import Concatenate 
from keras.models import Model
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input, VGG16
from sklearn.model_selection import train_test_split

import keras.backend

NUM_COMICS = 1248
NUM_PANELS = 3
MAX_SENTENCE_LEN = 60
BATCH_SIZE = 16

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
y = np.zeros((len(context_texts), len(words)), dtype=np.bool)

for index, data in tqdm(enumerate(zip(context_texts, next_words)),
                        desc='vectorize', total=len(next_words)):
    context, next_word = data
    for word_index, word in enumerate(context):
        x_text[index, word_index, word_to_index[word]] = 1
    y[index, word_to_index[next_word]] = 1

# TODO: REMOVE
x_text = x_text[:50]
y = y[:50]
panels = panels[:50]


split_data = train_test_split(x_text, y, panels, test_size=0.2, random_state=42)
x_test_train, x_test_val, y_train, y_val, panels_train, panels_val = split_data

def data_generator(is_val=False):
    gen_x_text = x_test_val if is_val else x_test_train
    gen_y = y_val if is_val else y_train
    gen_panels = panels_val if is_val else panels_train
    while True:
        for start_index in range(0, len(gen_panels), BATCH_SIZE):
            x_image_raw = np.zeros((min(BATCH_SIZE, 
                len(gen_panels) - start_index), 224, 224, 3))
            for index in range(BATCH_SIZE):
                try:
                    x_image_raw[index, :, :, :] = image.img_to_array(
                        gen_panels[start_index + index])
                except IndexError:
                    break

            batch_x_text = gen_x_text[start_index:start_index+BATCH_SIZE]
            batch_y = gen_y[start_index:start_index+BATCH_SIZE]
            batch_x_image = preprocess_input(x_image_raw)
            yield ([batch_x_text, batch_x_image], batch_y)


image_input = Input(shape=(224, 224, 3))
vgg = VGG16(include_top=False, weights='imagenet',
            input_tensor=image_input)
for layer in vgg.layers:
    layer.trainable = False
vgg_dense = Dense(128)(Flatten()(vgg.output))
vgg_embed = RepeatVector(1)(vgg_dense)

text_input = Input(shape=(MAX_SENTENCE_LEN, len(words)))
text_embed = TimeDistributed(Dense(128))(text_input)

lstm_input = Concatenate(axis=1)([vgg_embed, text_embed])
lstm = Bidirectional(LSTM(128), input_shape=(
    MAX_SENTENCE_LEN, len(words)))(lstm_input)
predictions = Dense(len(words), activation='softmax')(lstm)
model = Model(inputs=(text_input, image_input), outputs=predictions)
model.compile(optimizer='adam', loss='categorical_crossentropy')
model.fit_generator(data_generator(), 
                    steps_per_epoch=np.ceil(len(panels_train) / BATCH_SIZE),
                    epochs=10,
                    validation_data=data_generator(is_val=True),
                    validation_steps=np.ceil(len(panels_val) / BATCH_SIZE))
