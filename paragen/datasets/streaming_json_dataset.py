import json
import logging
logger = logging.getLogger(__name__)

from paragen.datasets import register_dataset
from paragen.datasets.streaming_dataset import StreamingDataset
from paragen.utils.io import UniIO


@register_dataset
class StreamingJsonDataset(StreamingDataset):
    """
    StreamingJsonDataset is a streaming dataset for reading data saved with json.dumps.

    Args:
        path: a dict of data with their path. `path` can be `None` to build the process pipeline only.
    """

    def __init__(self, path):
        super().__init__(path)

    def build(self, collate_fn=None, preprocessed=False):
        """
        Build input stream

        Args:
             collate_fn: callback defined by a specific task
             preprocessed: data has been processed
        """
        self._collate_fn = collate_fn
        self._preprocessed = preprocessed

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
            except Exception as e:
                logger.warning(e)

    def _callback(self, sample):
        """
        Callback for json data

        Args:
            sample: data in raw format

        Returns:
            sample (dict): a dict of samples consisting of parallel data of different sources
        """
        sample = json.loads(sample)
        return sample

    def reset(self):
        """
        reset the dataset
        """
        self._pos = 0
        self._fin = UniIO(self._path)

    def finalize(self):
        """
        Finalize dataset after finish reading
        """
        self._fin.close()
