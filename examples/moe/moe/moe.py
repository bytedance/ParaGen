import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from paragen.modules.utils import get_activation_fn


class MoE(nn.Module):
    """
    MoE layer.

    Args:
        d_model: input feature dimension
        dim_feedforward: dimensionality of inner vector space
        dim_out: output feature dimensionality
        activation: activation function
        bias: requires bias in output linear function
        num_experts: number of experts in MoE layer
            The layer degenerates to standard Transformer layer when num_experts=1
        sparse: whether using sparse routing (one token only passes one expert)
    """
    def __init__(self,
                 d_model,
                 dim_feedforward=None,
                 dim_out=None,
                 activation="relu",
                 num_experts=1,
                 sparse=True):
        super().__init__()
        
        self._d_model = d_model
        self._dim_feedforward = dim_feedforward or d_model
        self._dim_out = dim_out or d_model
        self._num_experts = num_experts
        self._sparse = sparse

        self._init_ffns()
        self._activation = get_activation_fn(activation)
        self._gate = nn.Linear(self._d_model, self._num_experts, bias=False)
    
    def _init_ffns(self):
        self._fc1_weight = nn.Parameter(torch.Tensor(self._num_experts, self._dim_feedforward, self._d_model))
        self._fc1_bias = nn.Parameter(torch.Tensor(self._num_experts, self._dim_feedforward))
        self._fc2_weight = nn.Parameter(torch.Tensor(self._num_experts, self._d_model, self._dim_feedforward))
        self._fc2_bias = nn.Parameter(torch.Tensor(self._num_experts, self._d_model))

        for i in range(self._num_experts):
            nn.init.kaiming_uniform_(self._fc1_weight[i], a=math.sqrt(5))
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self._fc1_weight[i])
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self._fc1_bias[i], -bound, bound)

            nn.init.kaiming_uniform_(self._fc2_weight[i], a=math.sqrt(5))
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self._fc2_weight[i])
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self._fc2_bias[i], -bound, bound)
    
    def forward(self, x):
        """
            B: batch size (sentence)
            L: sentence length
            d: embeddinng dim
            df: ffn dim
            n: number of experts

            x: [B,L,d]
            gate_out, expert_scores: [B,L,n]
            _fc1_weight: [n,df,d]
            _fc1_bias: [n,df]
            _fc2_weight: [n,d,df]
            _fc2_bais: [n,d]
            selected_importances, selected_experts: [B,L]

            Returns:
                - layer output and load balacing loss
        """
        gate_out = self._gate(x)
        expert_scores = F.softmax(gate_out, dim=-1)
        loss = 0.
        if self._sparse:
            nums_token = expert_scores.size(0)*expert_scores.size(1)
            selected_importances, selected_experts = torch.max(expert_scores, dim=-1)
            token_fraction = torch.zeros(self._num_experts).to(expert_scores)
            for idx in range(self._num_experts):
                mask = selected_experts.eq(idx)
                token_fraction[idx] = mask.sum()
                routed_input = x[mask]
                routed_importance = selected_importances[mask]
                if routed_input.size(0)!=0:
                    ffn_out = self.ffn(routed_input, idx, routed_importance)
                    x[mask] = ffn_out.to(x)
            token_fraction /= nums_token
            prob_fraction = expert_scores.sum(0).sum(0)/nums_token
            loss += torch.dot(token_fraction, prob_fraction)*self._num_experts
        else:
            x = torch.einsum("bld,nfd->blnf",[x, self._fc1_weight])
            x += self._fc1_bias
            x = self._activation(x)
            x = torch.einsum("blnf,ndf->blnd", [x, self._fc2_weight])
            x += self._fc2_bias
            x *= expert_scores.unsqueeze(-1)
            x = x.sum(dim=2)
        return x, loss

    def ffn(self, x, idx_expert, importance):
        dispatch_fc1_weight = self._fc1_weight[idx_expert]
        dispatch_fc1_bias = self._fc1_bias[idx_expert]
        x = F.linear(x, dispatch_fc1_weight, dispatch_fc1_bias)
        x = self._activation(x)
        dispatch_fc2_weight = self._fc2_weight[idx_expert]
        dispatch_fc2_bias = self._fc2_bias[idx_expert]
        x = F.linear(x, dispatch_fc2_weight, dispatch_fc2_bias)
        x *= importance.unsqueeze(-1)
        return x

