import csv
import glob
import os.path

from textgenrnn import textgenrnn

texts = []
text_glob = os.path.join('data', 'transcriptions', '*')
for text_path in glob.glob(text_glob):
    with open(text_path) as f:
        texts.extend(list(csv.reader(f))[0])

net = textgenrnn()
net.train_on_texts(texts, num_epochs=15)
net.save()
