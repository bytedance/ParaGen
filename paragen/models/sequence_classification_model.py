import torch

from paragen.models import AbstractModel, register_model
from paragen.modules.encoders import create_encoder
from paragen.modules.layers.embedding import Embedding
from paragen.modules.layers.classifier import HuggingfaceClassifier


@register_model
class SequenceClassificationModel(AbstractModel):
    """
    SequenceClassificationModel is a general sequence classification architecture consisting of
    one encoder and one classifier.

    Args:
        encoder: encoder configuration
        labels: number of labels
        dropout: dropout
        source_num: the number of input source sequence
        path: path to restore model
    """

    def __init__(self,
                 encoder,
                 labels,
                 dropout=0.,
                 source_num=1,
                 path=None):
        super().__init__(path)
        self._encoder_config = encoder
        self._labels = labels
        self._source_num = source_num
        self._dropout = dropout

        self._encoder, self._classifier = None, None
        self._path = path

    def _build(self, vocab_size, special_tokens):
        """
        Build model with vocabulary size and special tokens

        Args:
            vocab_size: vocabulary size of input sequence
            special_tokens: special tokens of input sequence
        """
        self._build_encoder(vocab_size=vocab_size, special_tokens=special_tokens)
        self._build_classifier()

    def _build_encoder(self, vocab_size, special_tokens):
        """
        Build encoder with vocabulary size and special tokens

        Args:
            vocab_size: vocabulary size of input sequence
            special_tokens: special tokens of input sequence
        """
        self._encoder = create_encoder(self._encoder_config)
        embed = Embedding(vocab_size=vocab_size,
                          d_model=self.encoder.d_model,
                          padding_idx=special_tokens['pad'])
        self._encoder.build(embed=embed, special_tokens=special_tokens)

    def _build_classifier(self):
        """
        Build classifer on label space
        """
        self._classifier = HuggingfaceClassifier(self.encoder.out_dim * self._source_num, self._labels, dropout=self._dropout)

    @property
    def encoder(self):
        return self._encoder

    @property
    def classifier(self):
        return self._classifier

    def forward(self, *inputs):
        """
        Compute output with neural input

        Args:
            *inputs: input source sequences

        Returns:
            - log probability of labels
        """
        x = [self.encoder(t)[-1] for t in inputs]
        x = torch.cat(x, dim=-1)
        logits = self.classifier(x)
        return logits

