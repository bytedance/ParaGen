import logging
logger = logging.getLogger(__name__)

from paragen.datasets import register_dataset
from paragen.datasets.streaming_dataset import StreamingDataset
from paragen.utils.io import UniIO


@register_dataset
class StreamingTextDataset(StreamingDataset):
    """
    StreamingTextDataset is a streaming dataset for reading data in textual format.

    Args:
        path: path to load the data
    """

    def __init__(self,
                 path,):
        super().__init__(path)

    def build(self, collate_fn=None, preprocessed=False):
        """
        Build input stream

        Args:
             collate_fn: callback defined by a specific task
        """
        self._collate_fn = collate_fn

        if self._path:
            self._fin = UniIO(self._path)

    def __iter__(self):
        """
        fetch next sample

        Returns:
            sample: next sample
        """
        for sample in self._fin:
            try:
                sample = self._full_callback(sample)
                yield sample
            except StopIteration:
                raise StopIteration
            except Exception as e:
                logger.warning(e)

    def reset(self):
        """
        reset the dataset
        """
        self._pos = 0
        self._fin = UniIO(self._path)

    def _callback(self, sample):
        """
        Callback for json data

        Args:
            sample: data in raw format

        Returns:
            sample (dict): a dict of samples consisting of parallel data of different sources
        """
        sample = sample.strip('\n').strip()
        return sample

    def finalize(self):
        """
        Finalize dataset after finish reading
        """
        self._fin.close()

