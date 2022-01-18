from paragen.modules.encoders.transformer_encoder import TransformerEncoder
from paragen.models.bert_model import init_bert_params
import torch
import unittest


class TestTransformerEncoder(unittest.TestCase):
    def test_bert_initialization(self):
        encoder = TransformerEncoder(3, d_model=512, n_head=4, dim_feedforward=512)
        embed = torch.nn.Embedding(100, 512)
        encoder.build(embed, None)
        encoder.apply(init_bert_params)

        self.assertAlmostEqual(encoder._embed.weight.std().item(), 0.02, places=4)
        self.assertAlmostEqual(encoder._layers[0].self_attn.in_proj_weight.std().item(), 0.02, places=4)
        self.assertAlmostEqual(encoder._layers[0].self_attn.out_proj.weight.std().item(), 0.02, places=4)
        self.assertAlmostEqual(encoder._layers[0].ffn._fc1.weight.std().item(), 0.02, places=4)



if __name__ == '__main__':
    unittest.main()
