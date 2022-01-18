from paragen.models import register_model
from paragen.models.seq2seq import Seq2Seq


@register_model
class VariationalAutoEncoders(Seq2Seq):
    """
    VariationalAutoEncoders is an extension to Seq2Seq model with latent space .

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
                 path=None,
                 ):
        super().__init__(encoder=encoder,
                         decoder=decoder,
                         d_model=d_model,
                         share_embedding=share_embedding,
                         path=path)

    def reg_loss(self):
        """
        Auto-Encoding regularization loss

        Returns:
            - KL loss between prior and posterior
        """
        return self.encoder.reg_loss()

    def nll(self, rec_loss, reg_losses, method="elbo"):
        """
        NLL loss

        Args:
            rec_loss: reconstruction loss
            reg_losses: regularization loss
            method: generation method

        Returns:
            - NLL loss
        """
        return self.encoder.nll(rec_loss, reg_losses, method)

