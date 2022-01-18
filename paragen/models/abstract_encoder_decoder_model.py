from paragen.models import AbstractModel


class AbstractEncoderDecoderModel(AbstractModel):
    """
    AbstractEncoderDecoderModel defines interface for encoder-decoder model.
    It must contains two attributes: encoder and decoder.
    """

    def __init__(self, path, *args, **kwargs):
        super().__init__(path)
        self._args = args
        self._kwargs = kwargs

        self._encoder, self._decoder = None, None

    @property
    def encoder(self):
        return self._encoder

    @property
    def decoder(self):
        return self._decoder
