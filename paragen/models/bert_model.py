import logging
logger = logging.getLogger(__name__)

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import MultiheadAttention

from paragen.models import AbstractModel, register_model
from paragen.modules.encoders import create_encoder
from paragen.modules.layers.embedding import Embedding
from paragen.modules.utils import get_activation_fn
from paragen.modules.layers.classifier import HuggingfaceClassifier
from paragen.modules.encoders.transformer_encoder import TransformerEncoder
from paragen.modules.utils import param_summary


def init_bert_params(module):
    def normal_(data):
        data.copy_(
            data.cpu().normal_(mean=0.0, std=0.02).to(data.device)
        )

    if isinstance(module, nn.Linear):
        normal_(module.weight.data)
        if module.bias is not None:
            module.bias.data.zero_()
    if isinstance(module, nn.Embedding):
        normal_(module.weight.data)
        if module.padding_idx is not None:
            module.weight.data[module.padding_idx].zero_()
    if isinstance(module, MultiheadAttention):
        normal_(module.in_proj_weight.data)
        normal_(module.out_proj.weight.data)


class BertLMHead(torch.nn.Module):

    def __init__(self, embed_dim, output_dim, activation_fn, weight=None):
        super().__init__()
        self.dense = torch.nn.Linear(embed_dim, embed_dim)
        self.activation_fn = get_activation_fn(activation_fn)
        self.layer_norm = torch.nn.LayerNorm(embed_dim)

        if weight is None:
            weight = torch.nn.Linear(embed_dim, output_dim, bias=False).weight
        self.weight = weight
        self.bias = torch.nn.Parameter(torch.zeros(output_dim))

    def forward(self, features, masked_tokens=None):
        # Only project the masked tokens while training, saves both memory and computation
        if masked_tokens is not None:
            features = features[masked_tokens, :]

        x = self.dense(features)
        x = self.activation_fn(x)
        x = self.layer_norm(x)
        # project back to size of vocabulary with bias
        x = F.linear(x, self.weight) + self.bias
        return x


@register_model
class BertModel(AbstractModel):
    def __init__(self,
                 encoder,
                 d_model,
                 path=None):
        super().__init__(path=path)
        self._encoder_config = encoder
        self._d_model = d_model
        self._path = path

    def _build(self, vocab_size, special_tokens):
        embed = Embedding(vocab_size=vocab_size,
                          d_model=self._d_model,
                          padding_idx=special_tokens['pad'])
        self._encoder = create_encoder(self._encoder_config)
        self._encoder.build(embed=embed, special_tokens=special_tokens)
        self._lm_head = BertLMHead(
            embed_dim=self._d_model, 
            output_dim=vocab_size, 
            activation_fn=self._encoder_config['activation'],
            weight=embed.weight
        )
        self.apply(init_bert_params)
        numel_train, numel_total = param_summary(self)
        logger.info(f"Summary (trainable/total parameters): {numel_train}M/{numel_total}M")

    def forward(self, src_tokens, mask, **kwargs):
        x, _ = self._encoder(src_tokens)
        scores = self._lm_head(x.permute(1, 0, 2), mask)
        logits = F.log_softmax(scores, dim=-1)
        return logits

    def loss(self, src_tokens, mask, tgt_tokens):
        logits = self.forward(src_tokens, mask)

        target = tgt_tokens[mask]
        ntokens = mask.sum()

        accu = (logits.argmax(1) == target).float().mean()
        loss = F.nll_loss(logits, target)

        logging_states = {
            "loss":     loss.data.item(),
            "ntokens":  ntokens.data.item(),
            "accu":     accu.data.item()
        }

        return loss, logging_states

    def reset(self, mode, *args, **kwargs):
        self._mode = mode
        self._encoder.reset(mode, *args, **kwargs)


@register_model
class BertModelForClassification(BertModel):
    def __init__(self,
                 encoder,
                 d_model,
                 num_labels,
                 path=None):
        super().__init__(encoder, d_model, path=path)
        self._num_labels = num_labels

    def _build(self, vocab_size, special_tokens):
        super()._build(vocab_size, special_tokens)
        assert isinstance(self._encoder, TransformerEncoder)
        self._classifier = HuggingfaceClassifier(self._encoder.d_model, self._num_labels, dropout=self._encoder._dropout)
        self._special_tokens = special_tokens

    def forward(self, input):
        x, _ = self._encoder(input)
        x = x.permute(1, 0, 2)[:, 0, :]
        logits = self._classifier(x)
        output = logits if self._num_labels > 1 else logits.squeeze(dim=-1)
        return output
