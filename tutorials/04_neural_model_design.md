Neural models in ParaGen are subclass of `torch.nn.Module`.
ParaGen use dual families, `model` and `generator`, to category neural models.
The `model` family are usually used in training stage, while the `generator` family are used in inference stage.
For a translation example, the base model is `Seq2Seq` model.
In training, at the target side, `Seq2Seq` decoder takes a shifted target as its input for parallel computation, 
which accelerates training speed.
But in inference, the `Seq2Seq` only take previous-decoded tokens as its input, and decode the next one.
It is usually combined with a greedy search or beam search algorithm to decode complete sequence.
Thus one model may act differently in training and in inference.
We use two dual categories to describe the neural models.

# Model in training

Similar to other class creation in ParaGen, the neural model is built by the co-operation between `__init__` and `build`
function.
As is shown in [01_arguments_specification](./01_arguments_specification.md), `__init__` is used to receive arguments
from yaml.
But the `build` function receives arguments from `Task` instance, such as vocabulary size.
The `__init__` stores configurations without instantiating its internal modules, whereas the `build` function do
internal module instantiation.
We show an example of building a Seq2Seq model.
```python
from paragen.models import register_model, AbstractModel
from paragen.modules.decoders import create_decoder
from paragen.modules.encoders import create_encoder
from paragen.modules.utils import create_source_target_modality


@register_model
class Seq2Seq(AbstractModel):

    def __init__(self,
                 encoder,
                 decoder,
                 d_model,
                 share_embedding=None,
                 path=None):
        super().__init__(path=path)
        self._encoder_config, self._decoder_config = encoder, decoder
        self._d_model = d_model
        self._share_embedding = share_embedding
        self._path = path
        self._encoder = None
        self._decoder = None

    def build(self, src_vocab_size, tgt_vocab_size, src_special_tokens, tgt_special_tokens):
        src_embed, tgt_embed, tgt_out_proj = create_source_target_modality(
            d_model=self._d_model,
            src_vocab_size=src_vocab_size,
            tgt_vocab_size=tgt_vocab_size,
            src_padding_idx=src_special_tokens['pad'],
            tgt_padding_idx=tgt_special_tokens['pad'],
            share_embedding=self._share_embedding
        )
        self._encoder = create_encoder(self._encoder_config)
        self._decoder = create_decoder(self._decoder_config)
        self._encoder.build(embed=src_embed, special_tokens=src_special_tokens)
        self._decoder.build(embed=tgt_embed,
                            special_tokens=tgt_special_tokens,
                            out_proj=tgt_out_proj)

```
We first store encoder and decoder configurations in `__init__` and then build model in `build` function.
In the `build` function, it build a embedding layer with respect to vocabulary size, which is passed from a `Task` instance.

One tips to build model is to use `_build` instead of `build`, because it is not required to tackle with cuda or model 
loading issues in `_build`.

# Generator in inference

`Generator` class is designed to rewrite `Model`, enable end-to-end inference for a neural model.
A typical example of `Generator` is `Seq2Seq` model with beam search algorithm.
Another example is wrapping a pointer network to perform span extraction class.
Besides, a `Generator` instance is exactly the exported `torch.nn.Module` for online serving.
Here is an example of `SequenceGenerator`.
```python
from typing import Dict

from paragen.generators import AbstractGenerator, register_generator
from paragen.modules.search import create_search


@register_generator
class SequenceGenerator(AbstractGenerator):

    def __init__(self,
                 search: Dict=None,
                 path=None):
        super().__init__(path)
        self._search_configs = search

        self._model = None
        self._encoder, self._search = None, None
        self._src_special_tokens, self._tgt_special_tokens = None, None

    def build_from_model(self, model, src_special_tokens, tgt_special_tokens):
        self._model = model
        self._encoder = model.encoder
        self._src_special_tokens, self._tgt_special_tokens = src_special_tokens, tgt_special_tokens

        self._search = create_search(self._search_configs)
        self._search.build(decoder=model.decoder,
                           bos=self._tgt_special_tokens['bos'],
                           eos=self._tgt_special_tokens['eos'],
                           pad=self._tgt_special_tokens['pad'])

    def forward(self, encoder, decoder, search=tuple()):
        encoder_output = self._encoder(*encoder)
        _, decoder_output = self._search(*decoder, *encoder_output, *search)
        return decoder_output
```
`SequenceGenerator` is build from the original neural model with auxiliary configurations (search configurations).
It keeps encoder and wraps the decoder with a specified search algorithm.
When calling an `SequenceGenerator` objective, `forward` function is called to search a full sequence.
