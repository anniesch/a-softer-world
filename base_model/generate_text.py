import os.path
import csv

from tqdm import tqdm
from textgenrnn import textgenrnn

net = textgenrnn()
net.load('textgenrnn_weights_saved.hdf5')
gen_dir = os.path.join('data', 'generated')
if not os.path.exists(gen_dir):
    os.makedirs(gen_dir)
for index in tqdm(range(100)):
    gen_name = 'text_{:04d}.csv'.format(index)
    gen_path = os.path.join(gen_dir, gen_name)
    lines = net.generate(
        3, temperature=0.4, return_as_list=True, max_gen_length=120)
    with open(gen_path, 'w') as f:
        writer = csv.writer(f)
        writer.writerow(lines)
