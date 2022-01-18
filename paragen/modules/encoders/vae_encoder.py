import torch
import torch.nn as nn

from paragen.modules.encoders import AbstractEncoder, register_encoder, create_encoder
from paragen.modules.layers.gaussian import Gaussian
from paragen.modules.utils import sample_from_gaussian, mean_pooling


@register_encoder
class VAEEncoder(AbstractEncoder):
    """
    VAEEncoder is a variational auto-encoding wrapper for an encoder

    Args:
        encoder: inner encoder configurations
        latent_size: dimensionality of latent space
        name: encoder name
    """

    def __init__(self,
                 encoder,
                 latent_size,
                 name=None):
        super().__init__(name)
        self._encoder_configs = encoder
        self._latent_size = latent_size

        self._padding_idx = None
        self._encoder = None
        self._gaussian = None
        self._out_proj = None
        self._mode = 'train'

    def build(self, embed, special_tokens):
        """
        Build computational modules.

        Args:
            embed: token embedding
            special_tokens: special tokens defined in vocabulary
        """
        self._encoder = create_encoder(self._encoder_configs)
        self._encoder.build(embed, special_tokens)

        self._gaussian = Gaussian(self.d_model, self._latent_size)
        self._out_proj = nn.Linear(self._latent_size, self.d_model)

    def reg_loss(self):
        """
        Auto-Encoding regularization loss

        Returns:
            - KL loss between prior and posterior
        """
        kl_loss = -0.5 * torch.sum((1 + self._states['mean'] - self._states['logvar'].pow(2) - self._states['logvar'].exp()), dim=1)
        return kl_loss

    def nll(self, rec_loss, reg_losses, method='elbo'):
        """
        NLL loss

        Args:
            rec_loss: reconstruction loss
            reg_losses: regularization loss
            method: generation method

        Returns:
            - NLL loss
        """
        if method == 'elbo':
            return rec_loss + reg_losses
        elif method == 'importance_sampling':
            raise NotImplementedError

    def _forward(self, src):
        r"""
        Args:
            src: tokens in src side.
              :math:`(N, S)` where N is the batch size, S is the source sequence length.

        Outputs:
            - source token hidden representation.
              :math:`(S, N, E)` where S is the source sequence length, N is the batch size,
              E is the embedding size.
        """
        if self._mode == 'sample':
            dis = torch.zeros((src.size(0), self._latent_size)), torch.zeros((src.size(0), self._latent_size))
        else:
            # Encoding
            memory, memory_padding_mask = self._encoder(src=src)
            # Posterior inference
            reps = mean_pooling(memory, memory_padding_mask)
            dis = self._gaussian(reps)
            self._states['mean'], self._states['logvar'] = dis

        sample = sample_from_gaussian(*dis)

        encoder_out = self._out_proj(sample).unsqueeze(0)
        return encoder_out, None

    @property
    def d_model(self):
        return self._encoder.d_model

    @property
    def out_dim(self):
        return self._encoder.out_dim

    def set_mode(self, mode):
        self._mode = mode
