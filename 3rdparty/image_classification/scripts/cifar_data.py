import os

import numpy as np
from PIL import Image


def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict


for split in ['train', 'test']:
    data = unpickle(f'cifar-100-python/{split}')
    keys = [_ for _ in data]
    names, labels, items = keys[0], keys[2], keys[4]
    for name, label, item in zip(data[names], data[labels], data[items]):
        name = name.decode('utf8')
        path = f'cifar/{split}/{label}'
        if not os.path.exists(path):
            os.makedirs(path)
        item = np.asarray(item).reshape((3, 32, 32)).transpose((1,2,0))
        im = Image.fromarray(item)
        path = f'{path}/{name}'
        im.save(path)
