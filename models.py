import math

import torch
from torch import nn

import utils
from modules import Encoder


class TextEncoder(nn.Module):
    def __init__(self,
                 ph_vocab=61,
                 bpe_vocab=20000,
                 hidden_s=192,
                 n_heads=4,
                 p_dropout=0.2,
                 filter_s=768,
                 kernel_s=5,
                 emb_s=192,
                 n_layers=4):
        super().__init__()
        self.hidden_channels = hidden_s

        self.phone_emb = nn.Embedding(ph_vocab, emb_s)

        self.token_emb = nn.Embedding(bpe_vocab, emb_s)

        # nn.init.normal_(self.emb.weight, 0.0, hidden_s ** -0.5)

        self.phone_encoder = Encoder(
            hidden_s,
            filter_s,
            n_heads,
            n_layers,
            kernel_s,
            p_dropout)

        self.token_encoder = Encoder(
            hidden_s,
            filter_s,
            n_heads,
            n_layers,
            kernel_s,
            p_dropout)

        self.fuse_layer = Encoder(
            hidden_s,
            filter_s,
            n_heads,
            n_layers=1,
            kernel_size=kernel_s,
            p_dropout=p_dropout
        )

    def forward(self, x_ph, x_ph_lengths, x_t, x_t_lengths, w_b, w_lengths, ph_w):
        x_ph = self.phone_emb(x_ph) * math.sqrt(self.hidden_channels)  # [b, t, h]
        x_t = self.token_emb(x_t) * math.sqrt(self.hidden_channels)

        x_ph = torch.transpose(x_ph, 1, -1)  # [b, h, t]
        x_t = torch.transpose(x_t, 1, -1)  # [b, h, t]

        x_ph_mask = torch.unsqueeze(utils.sequence_mask(x_ph_lengths, x_ph.size(2)), 1).to(x_ph.dtype)
        x_t_mask = torch.unsqueeze(utils.sequence_mask(x_t_lengths, x_t.size(2)), 1).to(x_t.dtype)

        x_ph = torch.transpose(self.phone_encoder(x_ph * x_ph_mask, x_ph_mask), 1, -1)
        x_t = torch.transpose(self.token_encoder(x_t * x_t_mask, x_t_mask), 1, -1)
        print(x_ph.shape, x_t.shape)

        wd_s = utils.word_level_pooling(x_t, x_t_lengths, w_b, w_lengths)
        print(wd_s.shape)
        w_p = utils.word_to_phones(wd_s, w_lengths, ph_w, x_ph.shape[1])
        print(w_p.shape)
        algn_w_p = torch.transpose(torch.add(x_ph, w_p), 1, -1)

        ph_t_enc = torch.transpose(self.fuse_layer(algn_w_p, x_ph_mask), 1, -1)

        return x_ph_mask.shape, algn_w_p.shape, ph_t_enc.shape



# cnt = 0
model = TextEncoder().to("cuda")
# for param in model.parameters():
#    cnt += param.numel()


x_ph = torch.ones((1, 200), dtype=torch.int).to("cuda")
x_length = torch.tensor([90], dtype=torch.int).to("cuda")
x_t = torch.ones((1, 50), dtype=torch.int).to("cuda")
x_t_length = torch.tensor([22], dtype=torch.int).to("cuda")
w_b = torch.tensor([[2, 2, 2, 2, 2, 2, 2, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1]], dtype=torch.int).to("cuda")
w_length = torch.tensor([14], dtype=torch.int).to("cuda")
ph_w = torch.tensor([[10, 10, 10, 10, 10, 10, 10, 2, 4, 4, 2, 2, 2, 4]], dtype=torch.int).to("cuda")

print(model(x_ph, x_length, x_t, x_t_length, w_b, w_length, ph_w))


