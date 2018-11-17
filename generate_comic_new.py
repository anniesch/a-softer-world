import os
import glob
import random
import textwrap
import math
import numpy as np
# import markovify
from pathlib import Path

import glob
from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont

import csv
from io import StringIO

# Positioning: in: positioning data from all the extraction
# out: locations (3 positions, 1 for each panel) of words for asw comic

# if the inputs are
# TEST = [(5, 5), (5, 5), (5, 5)]
# PANEL_COORDS = [(13, 38), (254, 35), (491, 37)]

################################################################
# READ IN POSITIONS AND TEXT FROM OTHER FILES
positions_dir = os.path.join('data', 'text_positions')  # SUBJECT TO CHANGE
text_dir = os.path.join('data', 'generated')  # SUBJECT TO CHANGE
index = 35 # test
text_csv_name = 'text_{:04d}.csv'.format(index)
location_csv_name = 'loc_{:04d}.csv'.format(index)
csv_file = open(os.path.join(positions_dir, location_csv_name), "r")
csv_file_text = open(os.path.join(text_dir, text_csv_name), "r")

reader = csv.reader(csv_file)
list_positions = [eval(i) for i in list(reader)[0]] # convert strings to tuples
print(list_positions)

reader = csv.reader(csv_file_text)
new_text = list(reader)[0]
print(new_text)

################################################################
# Positioning


def positioning(positions):
	n = len(positions)
	all_x = []
	all_y = []
	for x, y in positions:
		all_x.append(x)
		all_y.append(y)

	average_x = np.mean(all_x)
	average_y = np.mean(all_y)
	x_stddev = np.std(all_x)
	y_stddev = np.std(all_y)
	print(average_x, average_y, x_stddev, y_stddev)

	# Make random distribution centered around averages, choose randomly
	new_x = np.random.normal(average_x, x_stddev / 5) ##actually y
	new_y = np.random.normal(average_y - 140, y_stddev / 5) ##actually x

	return (int(new_x), int(new_y)) ##goes y (col) then x (row)


PANEL_COORDS = []
for i in range(3):
    new_position = positioning(list_positions)
    PANEL_COORDS.append(new_position)

print("PANEL_COORDS", PANEL_COORDS)

################################################################
# IMAGE GENERATION
# Background image: in: image dataset (images), tags
# out: image for one asw comic (return image name). The directory is images.
image_dir = Path(os.path.join('data', 'images'))


def get_image():

	random_filename = random.choice(list(image_dir.glob('*.jpg')))
	image = Image.open(random_filename)
	#image.show()
	#print(random_filename)
	return random_filename


new_imagename = get_image()


################################################################
# GENERATE NEW COMIC
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
        y = random.randint(0, height - crop_size - 1)
        crop = back_image.crop((x, y, x + crop_size, y + crop_size))
        resized = crop.resize(size)
        panels.append(resized)
    return panels


def add_text(comic_image):
    counter = 0
    for coords, size in zip(PANEL_COORDS, PANEL_DIMENSIONS):
        print("size", size)
        sentence = new_text[counter]
        counter += 1
        wrapped = textwrap.wrap(sentence, width=20)
        draw = ImageDraw.Draw(comic_image)
        font = ImageFont.truetype('loveletter.ttf', 13)
        print("coords", coords)
        y = coords[0] + size[0]
        x = coords[1] + size[2]
        print("XAND Y", x, y)
        for line in wrapped:
            width, height = font.getsize(line)
            draw.rectangle(((x, y), (x + width, y + height)), fill='white')
            draw.text((x, y), line, font=font, fill='black')
            y += height + 5


def generate_comic():
    comic_image = Image.open(template_path)
    panels = generate_panels()
    for panel, coords in zip(panels, PANEL_DIMENSIONS):
        print("HELLO", panel, coords)
        coords_panels = coords[2], coords[0]
        comic_image.paste(panel, coords_panels)
    add_text(comic_image)
    comic_image.show()


generate_comic()




