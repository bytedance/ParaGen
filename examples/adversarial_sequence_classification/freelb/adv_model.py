import logging
logger = logging.getLogger(__name__)

import torch
import torch.nn.functional as F

from transformers import AutoConfig, AutoModelForSequenceClassification

from paragen.models import register_model
from paragen.models.abstract_model import AbstractModel
from paragen.utils.runtime import Environment as E
from paragen.utils.profiling import ram



@register_model
class AdvHuggingfaceSequenceClassificationModel(AbstractModel):

    def __init__(self, pretrained_model, num_labels=2, init_mag=0.05, adv_lr=0.1, max_norm=0.06):
        super().__init__()
        self._pretrained_model = pretrained_model
        self._num_labels = num_labels

        self._config = None
        self._model = None
        self._padding_idx = None

        self._init_mag = init_mag
        self._adv_lr = adv_lr
        self._max_norm = max_norm

    def build(self, vocab_size, padding_idx):
        self._config = AutoConfig.from_pretrained(
            self._pretrained_model,
            num_labels=self._num_labels
        )
        self._model = AutoModelForSequenceClassification.from_pretrained(
            self._pretrained_model,
            config=self._config,
        )
        # print(">>> debug!"); import debugpy; debugpy.listen(("127.0.0.1", 5678)); debugpy.wait_for_client(); debugpy.breakpoint()
        if "roberta" in self._pretrained_model:
            self._embed_fn = self._model.roberta.embeddings.word_embeddings
        elif "bert" in self._pretrained_model:
            self._embed_fn = self._model.bert.embeddings.word_embeddings
        else:
            raise Exception("Only support bert/roberta")
        self._padding_idx = padding_idx

        e = E()
        if e.device.startswith('cuda'):
            logger.info('move model to {}'.format(e.device))
            self.cuda(e.device)

        logger.info('neural network architecture\n{}'.format([_ for _ in self.children()]))

    def forward(self, input):
        if ram.has_flag("adv_mode"):
            # https://github.com/zhuchen03/FreeLB/blob/eb2370161037f3b03c5ab1262fb455dc7da4d361/fairseq-RoBERTa/launch/FreeLB/sst-fp32-clip.sh#L58
            # https://github.com/zhuchen03/FreeLB/blob/eb2370161037f3b03c5ab1262fb455dc7da4d361/fairseq-RoBERTa/fairseq/tasks/sentence_prediction.py#L78
            if ram.read("adv_iter") == 0:
                embeds_init = self._embed_fn(input)
                input_mask = (input != 1).to(embeds_init)
                delta = torch.zeros_like(embeds_init).uniform_(-1, 1) * input_mask.unsqueeze(2)
                dims = input_mask.sum(1, keepdim=True) * embeds_init.size(-1)
                # init mag
                mag = self._init_mag / torch.sqrt(dims)
                delta = (delta * mag.view(-1, 1, 1)).detach()
            else:
                embeds_init = self._embed_fn(input)
                delta = ram.read("delta")
                delta_grad = delta.grad.detach()
                denorm = torch.norm(delta_grad.view(delta_grad.size(0), -1), dim=1).view(-1, 1, 1)

                # adv lr
                delta = (delta + self._adv_lr * delta_grad / denorm).detach()

                delta_norm = torch.norm(delta.view(delta.size(0), -1).float(), p=2, dim=1).to(embeds_init).detach()
                exceed_mask = (delta_norm > self._max_norm).to(embeds_init)
                delta = delta * (self._max_norm / delta_norm * exceed_mask + (1-exceed_mask)).view(-1, 1, 1).detach()

            delta.requires_grad_()
            output = self._model(inputs_embeds=embeds_init + delta, attention_mask=input.ne(self._padding_idx))
            output = F.log_softmax(output.logits, dim=-1) if self._num_labels > 1 else output.logits.squeeze(dim=-1)
            ram.write("delta", delta)
            return output
        else:
            output = self._model(input, attention_mask=input.ne(self._padding_idx))
            output = F.log_softmax(output.logits, dim=-1) if self._num_labels > 1 else output.logits.squeeze(dim=-1)
            return output

        
