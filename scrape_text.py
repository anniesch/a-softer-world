# from PIL import Image
import pytesseract
import argparse
import cv2
import os
import numpy as np
from tqdm import tqdm

comic_dir = os.path.join('data', 'comics')

N_COMICS = 1248
DIMENSION_THRESHOLD = 10
EXP_HEIGHT = 265
EXP_WIDTH = 720
PANEL_DIMENSIONS = [(37, 256, 15, 228),
					(37, 256, 248, 470),
					(37, 256, 492, 700)]
NUM_PANELS = 3

for index in tqdm(range(11, 12)): #tqdm(range(1, N_COMICS + 1)):
	# load image
	img_name = 'img_{:04d}.jpg'.format(index)
	image = cv2.imread(os.path.join(comic_dir, img_name))
	height, width, _ = image.shape
	if abs(height - EXP_HEIGHT) > DIMENSION_THRESHOLD or abs(width - EXP_WIDTH) > DIMENSION_THRESHOLD: continue

	# preprocess panels
	image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) #greyscale
	image = cv2.threshold(image, 0, 255,
		cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1] # thresholding

	# Apply dilation and erosion to remove some noise
	# kernel = np.ones((1, 1), np.uint8)
	# image = cv2.dilate(image, kernel, iterations=1)
	# image = cv2.erode(image, kernel, iterations=1)
	image = cv2.GaussianBlur(image, (5, 5), 0)

	panels = [
		cv2.resize(image[PANEL_DIMENSIONS[i][0]:PANEL_DIMENSIONS[i][1], PANEL_DIMENSIONS[i][2]:PANEL_DIMENSIONS[i][3]], 
			None, fx=1.5, fy=1.5, interpolation=cv2.INTER_CUBIC)
		for i in range(NUM_PANELS)
		]
	image = cv2.resize(image, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_CUBIC)
	# split into 3 images
	
	# run OCR on each image
	text_per_panel = []
	for i in range(NUM_PANELS):
		text = pytesseract.image_to_string(panels[i], config = '--psm 11')
		print(text)
		text_per_panel.append(text)
		cv2.imshow("Image" + str(i), panels[i])
	 
	# show the output images
	cv2.waitKey(0)

class ComicText:
	def __init__(text_per_panel=[], text_positions = [], npanels=3):
		self.text_per_panel = text_per_panel
		self.text_positions = text_positions
		self.npanels = npanels

	def get_text(self, panel_number):
		return self.text_per_panel[panel_number - 1]

	def get_aggregate_text(self):
		return '\n'.join(self.text_per_panel)

# how many comics with 3 panels?
'''
n = 0
for index in tqdm(range(1, N_COMICS + 1)):
	img_name = 'img_{:04d}.jpg'.format(index)
	image = cv2.imread(os.path.join(comic_dir, img_name))
	height, width, _ = image.shape
	if abs(height - EXP_HEIGHT) > DIMENSION_THRESHOLD or abs(width - EXP_WIDTH) > DIMENSION_THRESHOLD: continue
	n += 1
print(n)
'''