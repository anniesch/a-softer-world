import os
import glob
import random
import textwrap

import markovify
from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont

trans_dir = os.path.join('data', 'transcriptions')
back_dir = os.path.join('data', 'backgrounds')
template_path = os.path.join('data', 'template.png')
PANEL_COORDS = [(13, 38), (254, 35), (491, 37)]
PANEL_SIZES = [(214, 214), (210, 214), (211, 217)]


def train_markov_chain():
    texts = []
    for trans_file in glob.glob(os.path.join(trans_dir, '*')):
        with open(trans_file) as f:
            texts.append(f.read())
    return markovify.Text('\n'.join(texts))


def generate_panels():
    back_path = random.choice(glob.glob(os.path.join(back_dir, '*')))
    back_image = Image.open(back_path)
    width, height = back_image.size
    crop_size = int(0.75 * min([width, height]))
    panels = []
    for size in PANEL_SIZES:
        x = random.randint(0, width - crop_size - 1)
        y = random.randint(0, height- crop_size - 1)
        crop = back_image.crop((x, y, x + crop_size, y + crop_size))
        resized = crop.resize(size)
        panels.append(resized)
    return panels


def add_text(comic_image, text_model):
    for coords, size in zip(PANEL_COORDS, PANEL_SIZES):
        sentence = text_model.make_short_sentence(70)
        wrapped = textwrap.wrap(sentence, width=20)
        draw = ImageDraw.Draw(comic_image)
        font = ImageFont.truetype('loveletter.ttf', 13)
        x = coords[0] + random.randint(int(0.1 * size[0]), int(0.4 * size[0]))
        y = coords[1] + random.randint(int(0.1 * size[1]), int(0.8 * size[1]))
        for line in wrapped:
            width, height = font.getsize(line)
            draw.rectangle(((x, y), (x + width, y + height)), fill='white')
            draw.text((x, y), line, font=font, fill='black')
            y += height + 5


def generate_comic():
    text_model = train_markov_chain()
    comic_image = Image.open(template_path)
    panels = generate_panels()
    for panel, coords in zip(panels, PANEL_COORDS):
        comic_image.paste(panel, coords)
    add_text(comic_image, text_model)
    comic_image.show()


generate_comic()
