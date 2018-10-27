import os 
import pytesseract
from PIL import Image

N_COMICS = 1248
img_dir = os.path.join('data', 'comics')

for index in range(1, N_COMICS + 1):
    img_name = 'img_{:04d}.jpg'.format(index)
    img_path = os.path.join(img_dir, img_name)
    data = pytesseract.image_to_data(
        Image.open(img_path), config='--psm 11 --oem 0', lang='eng')
    print(data)
    print(type(data))
