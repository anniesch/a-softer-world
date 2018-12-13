import os
from pathlib import Path
import glob
import random
import textwrap
import math
import numpy
import markovify
from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont

##Background image: in: image dataset (images), tags
##out: image for one asw comic (return image name). The directory is images.
image_dir = Path(os.path.join('data', 'images'))


def get_image():

	random_filename = random.choice(list(image_dir.glob('*.jpg')))
	image = Image.open(random_filename)
	image.show()
	print(random_filename)
	return random_filename

get_image()

