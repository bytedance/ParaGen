# How to profile your code

## STEP 1

Add two lines to your code.

```python
from paragen.utils.profiling import profiler           # ---> Add this line

class TransformerEncoder(AbstractEncoder):
    ...

    def _forward(self, src: Tensor):
        ...
        for layer in self._layers:
            with profiler.timeit("encoder_layer"):   # ---> Add this line
                x = layer(x, src_key_padding_mask=src_padding_mask)
```

## STEP 2

Modify the `train.yaml`.

```yaml
task:
    ...
env:
    profiling_window: 50
```

## STEP 3

Run your code.

```
---------------------  ---------  -------------------  ---------------------                                                                                                       
name                   num calls  secs                 secs/call                                                                                                                   
*total                 50         20.014517068862915   0.4002903413772583                                                                                                          
backward               50         11.264406204223633   0.22528812408447266                                                                                                         
forward                50         7.867807149887085    0.1573561429977417                                                                                                          
forward.encoder_layer  600        0.5555515289306641   0.0009259192148844401                                                                                                       
optimizer              50         0.725677490234375    0.0145135498046875                                                                                                          
*rest                  -          0.15662622451782227  -                                                                                                                           
---------------------  ---------  -------------------  ---------------------
```
