from paragen.models import register_model
from paragen.models.encoder_decoder_model import EncoderDecoderModel


@register_model
class Seq2Seq(EncoderDecoderModel):
    """
    EncoderDecoderModel defines overall encoder-decoder architecture.

    Args:
        encoder: encoder configurations to build an encoder
        decoder: decoder configurations to build an decoder
        d_model: feature embedding
        share_embedding: how the embedding is share [all, decoder-input-output, None].
            `all` indicates that source embedding, target embedding and target
             output projection are the same.
            `decoder-input-output` indicates that only target embedding and target
             output projection are the same.
            `None` indicates that none of them are the same.
        path: path to restore model
    """

    def __init__(self,
                 encoder,
                 decoder,
                 d_model,
                 share_embedding=None,
                 path=None):
        super().__init__(encoder=encoder,
                         decoder=decoder,
                         d_model=d_model,
                         share_embedding=share_embedding,
                         path=path)

    def forward(self, src, tgt):
        """
        Compute output with neural input

        Args:
            src: source sequence
            tgt: previous tokens at target side, which is a time-shifted target sequence in training

        Returns:
            - log probability of next token at target side
        """
        memory, memory_padding_mask = self._encoder(src=src)
        logits = self._decoder(tgt=tgt,
                               memory=memory,
                               memory_padding_mask=memory_padding_mask)
        return logits
