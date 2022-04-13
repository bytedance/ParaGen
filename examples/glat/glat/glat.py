from paragen.models import register_model
from paragen.models.abstract_encoder_decoder_model import AbstractEncoderDecoderModel
from paragen.modules.decoders import create_decoder
from paragen.modules.encoders import create_encoder
from paragen.modules.layers.classifier import LinearClassifier
from paragen.modules.utils import create_source_target_modality, uniform_assignment, create_sequence
from paragen.utils.ops import local_seed


@register_model
class GLATModel(AbstractEncoderDecoderModel):
    """
    GLATModel is glancing transformer for non-auto regressive generation on sequence.

    Args:
        encoder: encoder configurations to build an encoder
        decoder: decoder configurations to build an decoder
        d_model: feature embedding
        max_output_length: maximum output sequence length
        share_embedding: how the embedding is share [all, decoder-input-output, None].
            `all` indicates that source embedding, target embedding and target
             output projection are the same.
            `decoder-input-output` indicates that only target embedding and target
             output projection are the same.
            `None` indicates that none of them are the same.
        decoder_input: decoder initial input computation method, [`encoder_mapping`, `unk`]
        path: path to restore model
    """

    def __init__(self,
                 encoder,
                 decoder,
                 d_model,
                 max_output_length=1024,
                 share_embedding=None,
                 decoder_input='uniform_copy',
                 path=None):
        super().__init__(path=path)
        self._encoder_config, self._decoder_config = encoder, decoder
        self._d_model = d_model
        self._max_output_length = max_output_length
        self._share_embedding = share_embedding
        self._decoder_input = decoder_input
        self._path = path

        self._encoder_embed, self._decoder_embed = None, None
        self._length_predictor = None
        self._src_special_tokens, self._tgt_special_tokens = None, None
        self._seed = 0

    def _build(self, src_vocab_size, tgt_vocab_size, src_special_tokens, tgt_special_tokens):
        """
        Build encoder-decoder model

        Args:
            src_vocab_size: vocabulary size at source sitde
            tgt_vocab_size: vocabulary size at target sitde
            src_special_tokens: special tokens in source vocabulary
            tgt_special_tokens: special tokens in target vocabulary
        """
        self._src_special_tokens, self._tgt_special_tokens = src_special_tokens, tgt_special_tokens
        src_embed, tgt_embed, tgt_out_proj = create_source_target_modality(d_model=self._d_model,
                                                                           src_vocab_size=src_vocab_size,
                                                                           tgt_vocab_size=tgt_vocab_size,
                                                                           src_padding_idx=src_special_tokens['pad'],
                                                                           tgt_padding_idx=tgt_special_tokens['pad'],
                                                                           share_embedding=self._share_embedding)
        self._encoder = create_encoder(self._encoder_config)
        self._decoder = create_decoder(self._decoder_config)

        self._encoder.build(embed=src_embed, special_tokens=src_special_tokens)
        self._decoder.build(embed=tgt_embed,
                            special_tokens=tgt_special_tokens,
                            out_proj=tgt_out_proj)

        self._encoder_embed, self._decoder_embed = src_embed, tgt_embed
        self._length_predictor = LinearClassifier(d_model=self._d_model,
                                                  labels=self._max_output_length,
                                                  invalid_classes=[0])

    def forward(self,
                src,
                tgt_padding_mask,
                target,
                fusing_target_mask):
        """
        Compute output with neural input

        Args:
            src: source sequence
            tgt_padding_mask: target padding mask size for generation
            target: gold target
            fusing_target_mask: fusing initial decoder input with gold target

        Returns:
            dict:
                - **token**: log probability of predicted tokens
                - **length**: log probability of length
        """
        with local_seed(self.seed):
            src_hidden, src_padding_mask, cls_token = self._encoder(src=src)

        length_logits = self._length_predictor(cls_token)
        decoder_embed = self.calc_decoder_input(src_padding_mask,
                                                tgt_padding_mask,
                                                source=src,
                                                target=target,
                                                fusing_target_mask=fusing_target_mask)
        with local_seed(self.seed):
            logits = self._decoder(tgt=decoder_embed,
                                   memory=src_hidden,
                                   tgt_padding_mask=tgt_padding_mask,
                                   memory_padding_mask=src_padding_mask)
        return {'token': logits, 'length': length_logits}

    def calc_decoder_input(self,
                           src_padding_mask,
                           tgt_padding_mask,
                           source=None,
                           target=None,
                           fusing_target_mask=None):
        """
        Compute decoder initial input

        Args:
            src_padding_mask: padding mask at source side
            tgt_padding_mask: padding mask at target side
            target: gold target
            fusing_target_mask: fusing initial decoder input with gold target

        Returns:
            - decoder initial input
        """
        if self._decoder_input == 'unk':
            decoder_input = create_sequence(tgt_padding_mask,
                                            self._tgt_special_tokens['unk'],
                                            pad_id=self._tgt_special_tokens['pad'])
            decoder_input[:, 0] = self._tgt_special_tokens['bos']
            length_tgt = ((~tgt_padding_mask).int()).sum(dim=-1)
            decoder_input.scatter_(1, length_tgt[:, None] - 1, self._tgt_special_tokens['eos'])
            decoder_embed = self._decoder_embed(decoder_input)
        elif self._decoder_input == 'uniform_copy':
            decoder_embed = uniform_copy(self._encoder_embed(source),
                                         src_padding_mask,
                                         tgt_padding_mask)
        else:
            raise NotImplementedError

        if target is not None and fusing_target_mask is not None:
            decoder_embed = fusion(self._decoder_embed(target),
                                   decoder_embed,
                                   fusing_target_mask)
        return decoder_embed

    def set_seed(self, seed):
        """
        Set random seed

        Args:
            seed: random seed
        """
        self._seed = seed

    @property
    def seed(self):
        return self._seed

    def reset(self, mode):
        """
        Switch mode and reset internal states

        Args:
            mode: running mode
        """
        self._mode = mode
        self._encoder.reset(mode)
        self._decoder.reset(mode)

    @property
    def encoder(self):
        return self._encoder

    @property
    def decoder(self):
        return self._decoder

    @property
    def length_predictor(self):
        return self._length_predictor


def uniform_copy(encoder_embed, encoder_padding_mask, decoder_padding_mask):
    """
    assign encoder embedding to decoder one

    Args:
        encoder_embed: encoder embeddings
            :math:`(N, S, E)` where N is batch size, S is source sequence and E is feature dimension
        encoder_padding_mask: padding mask of encoder input
            :math:`(N, S)` where N is batch size and S is source sequence
        decoder_padding_mask: padding mask of decoder input
            :math:`(N, T)` where N is batch size and T is target sequence

    Returns:
        - derived decoder embedding
            :math:`(N, T, E)` where N is batch size, T is target sequence and E is feature dimension
    """
    embed_dim = encoder_embed.size(dim=-1)
    index_t = uniform_assignment(encoder_padding_mask, decoder_padding_mask)
    index_t = index_t.unsqueeze(dim=-1).repeat(1, 1, embed_dim)
    decoder_embed = encoder_embed.gather(dim=1, index=index_t)
    return decoder_embed


def fusion(target_embed, decoder_embed, fusing_target_mask):
    """
    Fusion initial decoder input with gold target embedding according a fusing mask.
    Note two embedding matrices are of the same shape.

    Args:
        target_embed: feature matrix of gold target
        decoder_embed: feature matrix of decoder input
        fusing_target_mask: fusing mask,
            where `1` indicates keeping target and `0` indicates keep decoder initial input.

    Returns:

    """
    embed_dim = target_embed.size(dim=-1)
    fusing_target_mask = fusing_target_mask.unsqueeze(dim=-1).repeat(1, 1, embed_dim)
    decoder_embed = decoder_embed.masked_fill(fusing_target_mask, 0) + target_embed.masked_fill(~fusing_target_mask, 0)
    return decoder_embed
