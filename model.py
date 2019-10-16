# -*- coding: utf-8 -*-

import torch
import torch.nn as nn


class AttnHrDualEncoderModel(nn.Module):
    def __init__(self, hidden_dim, vocab_size, embedding_dim, pretrained_embeds=None):
        super().__init__()

        if pretrained_embeds is not None:
            self.word_embeds = nn.Embedding.from_pretrained(pretrained_embeds)
        else:
            self.word_embeds = nn.Embedding(vocab_size, embedding_dim)

        self.headline_encoder = nn.GRU(embedding_dim, hidden_dim)
        self.paragraph_encoder = nn.GRU(embedding_dim, hidden_dim)

        bidirectional = True
        self.body_encoder = nn.GRU(hidden_dim, hidden_dim, bidirectional=bidirectional)
        self.bilinear = nn.Bilinear(hidden_dim, hidden_dim * (2 if bidirectional else 1), 1)

    def forward(self, inputs):
        # Note
        #  - GRU inputs: input [seq_len, batch, input_size], h_0 [num_layers * num_directions, batch, hidden_size]
        #  - GRU outputs: output [seq_len, batch, num_directions * hidden_size], h_n [num_layers * num_directions, batch, hidden_size]

        idx_headline, idx_body = inputs  # [N=64, L_headline=25] [N * L_para=50, L_sent=200]
        batch_size = len(idx_headline)
        x_headline = self.word_embeds(idx_headline.t())  # [L_headline, N, H_embed]
        x_body = self.word_embeds(idx_body.t())  # [L_sent, N * L_para, H_embed]

        _, h_headline = self.headline_encoder(x_headline)  # _, [1, N, H_enc]
        h_headline = h_headline.squeeze()  # [N, H_enc]

        h_paras = []
        for x_sent in x_body.split(batch_size, dim=1):  # [L_sent, N, H_embed]
            _, h_para= self.paragraph_encoder(x_sent)  # _, [1, N, H_enc]
            h_paras.append(h_para.squeeze())
        h_paras = torch.stack(h_paras, dim=0)  # [L_para, N, H_enc]

        _, h_body = self.body_encoder(h_paras)  # _, [2, N, H_enc]
        h_body = h_body.transpose(0, 1).reshape([batch_size, -1])  # [N, 2 * H_enc]

        return self.bilinear(h_headline, h_body)
