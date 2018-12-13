import cv2
import numpy as np


def belongs_to(contour, cluster):
    for other in cluster:
        x1, y1, w1, h1 = cv2.boundingRect(contour)
        x2, y2, w2, h2 = cv2.boundingRect(other)
        x_match = (abs(x2 - (x1 + w1)) < 40 or abs(x1 - (x2 + w2)) < 40)
        y_match = (abs(y1 - y2) < 20)
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


img = cv2.imread('end.jpg')
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
lower, upper = np.array([0, 0, 0]), np.array([180, 255, 100])
mask = cv2.inRange(hsv, lower, upper)
_, contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

letter_contours = []
for contour in contours:
    _, _, width, height = cv2.boundingRect(contour)
    if 4 <= width <= 10 and 6 <= height <= 10:
        letter_contours.append(contour)

clusters = []
for contour in letter_contours:
    for cluster in clusters:
        if belongs_to(contour, cluster):
            cluster.append(contour)
            break
    else:
        clusters.append([contour])

text_clusters = [cluster for cluster in clusters
                 if cluster_box(cluster)[2] > 70]
text_boxes = [cluster_box(cluster, margin=10) for cluster in text_clusters]

for text_box in text_boxes:
    x, y, w, h = text_box
    cv2.imshow('text', img[y:y+h, x:x+w])
    cv2.waitKey(0)
    cv2.destroyAllWindows()