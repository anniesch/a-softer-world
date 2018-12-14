import re
import os.path
import urllib.request

from lxml import etree
from tqdm import tqdm

N_COMICS = 1248
url_format = 'http://www.asofterworld.com/index.php?id={}'
img_url_format = 'http://www.asofterworld.com/clean/{}'
img_regex = (u'<img.*src="http://www.asofterworld.com/clean/(.*\.jpg)".*'
             u'title="(.*)".*onclick=.*>')
comic_dir = os.path.join('data', 'comics')

if not os.path.exists(comic_dir):
    os.makedirs(comic_dir)

for index in tqdm(range(1, N_COMICS + 1)):
    url = url_format.format(index)
    response = urllib.request.urlopen(url)
    html = response.read().decode('utf-8', 'ignore')
    image_name, alt_text = re.findall(img_regex, html, re.DOTALL)[0]
    img_url = img_url_format.format(image_name)
    img_response = urllib.request.urlopen(img_url)
    img_data = img_response.read()

    img_name = 'img_{:04d}.jpg'.format(index)
    alt_name = 'img_{:04d}.label'.format(index)
    with open(os.path.join(comic_dir, img_name), 'wb') as f:
        f.write(img_data)
    with open(os.path.join(comic_dir, alt_name), 'w') as f:
        f.write(alt_text)


