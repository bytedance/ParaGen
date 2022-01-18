ParaGen supports `torch.jit` and `LightSeq` to export a complete neural model in inference `Generator`.

The `export` function is called by `paragen-export`.
The objective of `export` is `Generator`, within which the export codes are implemented.

Usually, to export a neural model, the `yaml` file of evaluation is directly used and specifes `export.path`
of exported model.
```bash
paragen-export --config {EVAL_CONFIG_PATH} --export.path {OUTPUT_PATH}
```

# Export & Load

Here is an example of exporting a `SequenceGenerator`:
```python
import torch

from paragen.generators import AbstractGenerator, register_generator
from paragen.modules.search import create_search
from paragen.utils.tensor import to_device


@register_generator
class SequenceGenerator(AbstractGenerator):

    def export(self, path, net_input, *args, **kwargs):
        self.eval()
        self.reset('infer')
        net_input = to_device(net_input, device=self._env.device)
        with torch.no_grad():
            encoder = torch.jit.trace_module(self._encoder, {'forward': net_input['encoder']})
            encoder_out = encoder(*net_input['encoder'])
            decoder = torch.jit.trace_module(self._model.decoder, {'forward': net_input['decoder'] + encoder_out})
            self._search_configs['class'] = self._search.__class__.__name__
            search = create_search(self._search_configs)
            search.build(decoder=decoder,
                         bos=self._tgt_special_tokens['bos'],
                         eos=self._tgt_special_tokens['eos'],
                         pad=self._tgt_special_tokens['pad'])
            search = torch.jit.script(search)
        with open(f'{path}/encoder', 'wb') as fout:
            torch.jit.save(encoder, fout)
        with open(f'{path}/search', 'wb') as fout:
            torch.jit.save(search, fout)

    def load(self):
        with open(f'{self._path}/encoder', 'rb') as fin:
            self._encoder = torch.jit.load(fin)
        with open(f'{self._path}/search', 'rb') as fin:
            self._search = torch.jit.load(fin)
```

To save exported model, it is allowed to save a model all in a file or save each part of a model separately.
In the `SequenceGenerator` case, we first trace encoder and decoder of an encoder-decoder model separately.
Note that it is important to implements a torch module in a traceable way.
Then we re-build the search algorithm with traced decoder, and script the search algoithm together with traced decoder.
Finally, two main components are saved separately.
The `load` function is much more easy and is implemented in consistent with `export` one.


# LightSeq

LightSeq support export for limited models.
In ParaGen, only two types of neural models can be exported with LightSeq:
- standard transformer
- encoder-decoder model, where encoders are any type and decoders is a transformer.

Thus we recommend to use LightSeq as a blackbox. 
More details can be found at [LightSeq github](https://github.com/bytedance/lightseq).

You may follow this `yaml` file to export above models with LightSeq.
```yaml
task:
  ...
export:
  path: transformer.pb
  beam_size: 4
  length_penalty: 0.6
  extra_decode_length: 50
  generation_method: beam_search
  topk: 1
  topp: 0.75
  diverse_lambda: 0
  lang: de
  pad_shift: 1
```

