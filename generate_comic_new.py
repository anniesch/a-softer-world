import os
import glob
import random
import textwrap
import math
import numpy as np
import markovify
from pathlib import Path

import glob
import markovify
from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont

import csv
from io import StringIO

##Positioning: in: positioning data from all the extraction
##out: locations (3 positions, 1 for each panel) of words for asw comic

##if the inputs are 
#TEST = [(5, 5), (5, 5), (5, 5)]
#PANEL_COORDS = [(13, 38), (254, 35), (491, 37)]

################################################################
##READ IN POSITIONS AND TEXT FROM OTHER FILES
csv_file = open(positions.csv, "r")
csv_file_text = open(text.csv, "r")

reader = csv.reader(csv_file)
list_positions = list(reader)
print(list_positions)

reader = csv.reader(csv_file_text)
new_text = list(reader)
print(new_text)

################################################################
##Positioning
def positioning(positions):
	n = len(positions)
	all_x = []
	all_y = []
	for x,y in positions:
		all_x.append(x)
		all_y.append(y)
	
	average_x = np.mean(all_x)
	average_y = np.mean(all_y)
	x_stddev = np.std(all_x)
	y_stddev = np.std(all_y)
	#print(average_x, average_y, x_stddev, y_stddev)

	##Make random distribution centered around averages, choose randomly
	new_x = np.random.normal(average_x, x_stddev)
	new_y = np.random.normal(average_y, y_stddev)
	return (new_x, new_y)

new_positions = positioning(list_positions)
PANEL_COORDS = new_positions

################################################################
##IMAGE GENERATION
##Background image: in: image dataset (images), tags
##out: image for one asw comic (return image name). The directory is images.
image_dir = Path(os.path.join('data', 'images'))

def get_image():

	random_filename = random.choice(list(image_dir.glob('*.jpg')))
	image = Image.open(random_filename)
	image.show()
	print(random_filename)
	return random_filename

new_imagename = get_image()


################################################################
####GENERATE NEW COMIC
comic_dir = os.path.join('data', 'comics')

N_COMICS = 1248
DIMENSION_THRESHOLD = 10
EXP_HEIGHT = 265
EXP_WIDTH = 720
PANEL_DIMENSIONS = [(37, 256, 15, 228),
                    (37, 256, 248, 470),
                    (37, 256, 492, 700)]
NUM_PANELS = 3
PANEL_SIZES = [(214, 214), (210, 214), (211, 217)]

trans_dir = os.path.join('data', 'transcriptions')
back_dir = os.path.join('data', 'backgrounds')
template_path = os.path.join('data', 'template.png')


def generate_panels():
    back_path = new_imagename
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


def add_text(comic_image):
	counter = 0
    for coords, size in zip(PANEL_COORDS, PANEL_SIZES):
        sentence = new_text[counter]
        counter += 1
        wrapped = textwrap.wrap(sentence, width=20)
        draw = ImageDraw.Draw(comic_image)
        font = ImageFont.truetype('loveletter.ttf', 13)
        x = coords[0]
        y = coords[1]
        for line in wrapped:
            width, height = font.getsize(line)
            draw.rectangle(((x, y), (x + width, y + height)), fill='white')
            draw.text((x, y), line, font=font, fill='black')
            y += height + 5


def generate_comic():
    #text_model = train_markov_chain()
    comic_image = Image.open(template_path)
    panels = generate_panels()
    for panel, coords in zip(panels, PANEL_COORDS):
        comic_image.paste(panel, coords)
    add_text(comic_image)
    comic_image.show()


generate_comic()




