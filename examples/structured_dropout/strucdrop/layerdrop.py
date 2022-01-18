import torch
import torch.nn as nn
import math


class LayerDropModuleList(nn.ModuleList):
    """
    Implementation of LayerDrop based on torch.nn.ModuleList

    Usage:
        Replace torch.nn.ModuleList with LayerDropModuleList
        For example:
            layers = nn.ModuleList([TransformerEncoderLayer 
                                    for _ in range(num_layers)])
                                    
        ->  layers = LayerDropModuleList(p=0.2, gamma=1e-4)
            layers.extend([TransformerEncoderLayer 
                           for _ in range(num_layers)])

    Args:
        p: initial drop probability 
        gamma: attenuation speed of drop probability
        mode_depth: probability distribution across layers
        modules: an iterable of modules to add
    """

    def __init__(self, p, gamma=0., mode_depth=None, modules=None):
        super().__init__(modules)
        self.p = p
        self._gamma = gamma
        self._mode_depth = mode_depth
        self._step = -1

    def __iter__(self):
        self._step += 1
        layer_num = len(self)
        dropout_probs = torch.empty(layer_num).uniform_()
        if self.training and self._gamma > 0:
            p_now = self.p - self.p * math.exp(-self._gamma * self._step) 
        else:
            p_now = self.p
        p_now = max(0., p_now)

        p_layers = [p_now] * layer_num
        if self._mode_depth == 'transformer':
            p_layers = [2*min(i+1, layer_num-i)/layer_num*p_now for i in range(layer_num)]
        elif self._mode_depth == 'bert':
            p_layers = [p_now*i/layer_num for i in range(1, layer_num+1)]

        for i, m in enumerate(super().__iter__()):
            m.layerdrop = p_layers[i] if self.training else 0.
            if not self.training or (dropout_probs[i] > m.layerdrop):
                yield m


def config_to_params(config):
    layerdrop, gamma, mode_depth = 0., 0., None
    if config is None:
        return layerdrop, gamma, mode_depth
    if 'prob' in config:
        layerdrop = config['prob']
    if 'gamma' in config:
        gamma = config['gamma']
    if 'mode_depth' in config:
        mode_depth = config['mode_depth']
    return layerdrop, gamma, mode_depth
