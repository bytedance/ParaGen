import os
import logging
logger = logging.getLogger(__name__)

from PIL import Image
import numpy as np

from paragen.datasets import register_dataset
from paragen.datasets.in_memory_dataset import InMemoryDataset
from paragen.utils.runtime import progress_bar


@register_dataset
class ImageDataset(InMemoryDataset):

    def __init__(self,
                 path,
                 sort_samples=False,
                 max_size=0):
        super().__init__(path, sort_samples=sort_samples, max_size=max_size)

    def _load(self):
        """
        Preload all the data into memory. In the loading process, data are preprocess and sorted.
        """
        image_label_pairs = []
        accecpted, discarded = 0, 0
        for label in os.listdir(self._path):
            path = os.path.join(self._path, label)
            for file in os.listdir(path):
                image_label_pairs.append((os.path.join(path, file), label))
        self._data = []
        for i, sample in enumerate(progress_bar(image_label_pairs, streaming=True, desc='Loading Samples')):
            if 0 < self._max_size <= i:
                break
            try:
                self._data.append(self._full_callback(sample))
                accecpted += 1
            except Exception:
                logger.warning('sample {} is discarded'.format(sample))
                discarded += 1
        self._length = len(self._data)
        logger.info(f'Totally accept {accecpted} samples, discard {discarded} samples')

    def _callback(self, sample):
        image, label = sample
        return {
            'image': Image.open(image),
            'label': eval(label)
        }


if __name__ == '__main__':
    dataset = ImageDataset('../data/cifar/test/', max_size=10)
    dataset.build()
    for x in dataset:
        print(x)
