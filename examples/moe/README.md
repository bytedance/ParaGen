# Mixture of Experts (MoE)

Replacing FFN layer with MoE layer can effectively enlarge the model size of Transformer model to achieve performance enhancement.
Sparse routing mechanism keeps the FLOPS approximately invariant while increasing the parameter count.

# Training by MoE model with 4 experts

On Wmt-14 ende

First prepare vocab and dataset in folder 'data', then run:

'''
bash run.sh -lib examples/moe/moe -config examples/moe/configs/train.yaml
'''
