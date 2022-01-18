ParaGen support training speedup by LightSeq. 
Currently, we can use `LSAdam` and `LSLabelSmoothedCrossEntropy` to speedup training in a model-agnostic way.
Besides, we also provides `LSTransformerEncoder` and `LSTransformerDecoder` to speed up the \
training process of a Transformer model.
Note that all the `LS` components are modularized and can be used individually.
Besides, they match any existing ParaGen component.

For example, we can train a lightseq-optimized transformer by setting training configuration as 
```yaml
task:
  class: ...
  model:
    class: Seq2Seq
    encoder:
      class: LSTransformerEncoder # TransformerEncoder in non-optimized version
      ...
    decoder:
      class: LSTransformerDecoder # TransformerDecoder in non-optimized version
      ...
    ...
  criterion:
    class: LSLabelSmoothedCrossEntropy # LabelSmoothedCrossEntropy in non-optimized version
    ...
  trainer:
    class: Trainer
    optimizer:
      class: LSAdam # Adam in non-optimized version
      ...
    ...
  ...
```
All the thing to use LightSeq training is simply add a prefix `LS` to class name.

# Bridge LightSeq Training and LightSeq Inference

Unlike a vanilla transformer, a transformer trained with LightSeq can not use exact the same configuration as \
[Export Tutorial](./07_model_export.md).
Because the lighseq transformer has a completely different state dictionary from the vanilla one.
But it is also easy to use ParaGen to export a LightSeq transformer by add a prefix `LS` to `LightseqTransformerGenerator`, \
namely `LSLightseqTransformerGenerator`.
The export yaml file is shown as follows
```yaml
task:
  class: ...
  model:
    class: Seq2Seq
    encoder:
      class: LSTransformerEncoder # TransformerEncoder in non-optimized version
      ...
    decoder:
      class: LSTransformerDecoder # TransformerDecoder in non-optimized version
      ...
    ...
  generator:
    class: LSLightseqTransformerGenerator
  ...
```

Note that for the evaluation in the training process, the LightSeq transformer is also suitable to `SequenceGenerator` \
and is not required to export its lightseq(-inference) version.
