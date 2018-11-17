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


def get_text_boxes(img):
    def belongs_to(contour, cluster):
        cluster_y = []
        for other in cluster:
            _, y, _, h = cv2.boundingRect(other)
            cluster_y.append(y + h/2.0)
        midline = sum(cluster_y) / len(cluster_y)
        for other in cluster:
            x1, y1, w1, h1 = cv2.boundingRect(contour)
            x2, y2, w2, h2 = cv2.boundingRect(other)
            x_match = (abs(x2 - (x1 + w1)) < 20 or abs(x1 - (x2 + w2)) < 20)
            y_match = (abs(y1 + h1 / 2.0 - midline) < 5)
            if x_match and y_match:
                return True
        return False

    def cluster_box(cluster, margin=0):
        x1s, x2s, y1s, y2s = [], [], [], []
        for contour in cluster:
            x, y, w, h = cv2.boundingRect(contour)
            x1s.append(x)
            x2s.append(x + w)
            y1s.append(y)
            y2s.append(y + h)
        new_x1, new_x2 = min(x1s), max(x2s)
        new_y1, new_y2 = min(y1s), max(y2s)
        new_w, new_h = new_x2 - new_x1, new_y2 - new_y1
        return (new_x1 - margin, new_y1 - margin,
                new_w + 2 * margin, new_h + 2 * margin)


    def is_white(img):
        lower, upper = np.array([0, 0, 200]), np.array([180, 255, 255])
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, lower, upper)
        return np.sum(mask == 255) / mask.size > 0.2

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower, upper = np.array([0, 0, 0]), np.array([180, 255, 100])
    mask = cv2.inRange(hsv, lower, upper)
    _, contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    letter_contours_x = []
    for contour in contours:
        x, y, width, height = cv2.boundingRect(contour)
        if 4 <= width <= 60 and 6 <= height <= 60:
            letter_contours_x.append((x, contour))

    letter_contours = [contour for _, contour in sorted(
        letter_contours_x, key=lambda pair: pair[0])]

    clusters = []
    for contour in letter_contours:
        for cluster in clusters:
            if belongs_to(contour, cluster):
                cluster.append(contour)
                break
        else:
            clusters.append([contour])

    text_clusters = []
    for cluster in clusters:
        x, y, w, h = cluster_box(cluster)
        if is_white(img[y:y+h, x:x+w]) and w > 50:
            text_clusters.append(cluster)

    text_boxes = [cluster_box(cluster, margin=5) for cluster in text_clusters]
    for text_box in text_boxes:
        x, y, w, h = text_box
        if x < 0: x = 0
        if y < 0: y = 0
    return text_boxes

for index in tqdm(range(7, 8)): #tqdm(range(1, N_COMICS + 1)):
	# load image
	img_name = 'img_{:04d}.jpg'.format(index)
	image = cv2.imread(os.path.join(comic_dir, img_name))
	height, width, _ = image.shape
	if abs(height - EXP_HEIGHT) > DIMENSION_THRESHOLD or abs(width - EXP_WIDTH) > DIMENSION_THRESHOLD: continue

	# preprocess panels
	# image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) #greyscale
	# image = cv2.threshold(image, 0, 255,
		# cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1] # thresholding

	# Apply dilation and erosion to remove some noise
	# kernel = np.ones((1, 1), np.uint8)
	# image = cv2.dilate(image, kernel, iterations=1)
	# image = cv2.erode(image, kernel, iterations=1)
	# image = cv2.GaussianBlur(image, (5, 5), 0)

	panels = [
		cv2.resize(image[PANEL_DIMENSIONS[i][0]:PANEL_DIMENSIONS[i][1], PANEL_DIMENSIONS[i][2]:PANEL_DIMENSIONS[i][3]], 
			None, fx=1.5, fy=1.5, interpolation=cv2.INTER_CUBIC)
		for i in range(NUM_PANELS)
		]
	# image = cv2.resize(image, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_CUBIC)
	# split into 3 images
	
	# run OCR on each image
	text_per_panel = []
	for i in range(NUM_PANELS):
		panel = panels[i]
		text_boxes = get_text_boxes(panel)
		# cv2.imshow("Image" + str(i), panels[i])
		for text_box in text_boxes:
		    x, y, w, h = text_box
		    print(text_box)
		    text = pytesseract.image_to_string(panel[y:y+h, x:x+w], config = '--psm 11')
		    print(text)
		    cv2.imshow('Image' + str(i) + ': text', panel[y:y+h, x:x+w])
		    cv2.waitKey(0)
		    cv2.destroyAllWindows()
		    text_per_panel.append(text)
	 
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
