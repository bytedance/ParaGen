import torch.nn as nn


class Gaussian(nn.Module):
    """
    Gaussian predict gaussian characteristics, namely mean and logvar

    Args:
        d_model: feature dimension
        latent_size: dimensionality of gaussian distribution
    """

    def __init__(self, d_model, latent_size):
        super().__init__()
        self._dmodel = d_model
        self._latent_size = latent_size

        self.post_mean = nn.Linear(d_model, latent_size)
        self.post_logvar = nn.Linear(d_model, latent_size)

    def forward(self, x):
        """
        Args:
            x: feature to perform gaussian
                :math:`(*, D)`, where D is feature dimension

        Returns:
            - gaussian mean
                :math:`(*, D)`, where D is feature dimension
            - gaussian logvar
                :math:`(*, D)`, where D is feature dimension
        """
        post_mean = self.post_mean(x)
        post_logvar = self.post_logvar(x)
        return post_mean, post_logvar


