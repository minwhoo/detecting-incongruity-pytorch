# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_sequence, pad_packed_sequence


class ParaHeadlineAttention(nn.Module):
    def __init__(self, para_dim, headline_dim, hidden_dim):
        super().__init__()
        self.linear_para = nn.Linear(para_dim, hidden_dim, bias=False)
        self.linear_headline = nn.Linear(headline_dim, hidden_dim, bias=False)
        self.v = nn.Parameter(torch.rand(hidden_dim))  # [H_hidden]

    def forward(self, paras, para_mask, headline):
        # paras: [N, P, H_para]
        # para_mask: [N, P]
        # headline: [N, H_headline]
        z = self.linear_para(paras) + self.linear_headline(headline).unsqueeze(1)  # [N, P, H_hidden]
        s = torch.tanh(z).matmul(self.v)  # [N, P]
        s[para_mask] = float('-inf')
        a = torch.softmax(s, dim=1)  # [N, P]
        return torch.matmul(a.unsqueeze(1), paras).squeeze()  # paras: [N, H_para]


class AttnHrDualEncoderModel(nn.Module):
    def __init__(self, hidden_dims, vocab_size, embedding_dim, pretrained_embeds=None):
        super().__init__()

        if pretrained_embeds is not None:
            self.word_embeds = nn.Embedding.from_pretrained(pretrained_embeds)
        else:
            self.word_embeds = nn.Embedding(vocab_size, embedding_dim)

        self.headline_encoder = nn.GRU(embedding_dim, hidden_dims['headline'])
        self.paragraph_encoder = nn.GRU(embedding_dim, hidden_dims['word'])

        self.body_encoder = nn.GRU(hidden_dims['word'], hidden_dims['paragraph'], bidirectional=True)
        self.attention = ParaHeadlineAttention(2 * hidden_dims['paragraph'], hidden_dims['headline'], 
                                               2 * hidden_dims['paragraph'])
        self.bilinear = nn.Bilinear(hidden_dims['headline'], 2 * hidden_dims['paragraph'], 1)

    # def forward(self, inputs):
    def forward(self, headlines, headline_lengths, bodys, para_lengths):
        # headline: [N, L_headline], 
        # headline_lengths: [N],
        # bodys: [N, P, L_para], 
        # para_lengths: [N, P]
        #
        # Note
        #  - GRU inputs: input [seq_len, batch, input_size], h_0 [num_layers * num_directions, batch, hidden_size]
        #  - GRU outputs: output [seq_len, batch, num_directions * hidden_size], h_n [num_layers * num_directions, batch, hidden_size]

        batch_size = len(headlines)  # N
        x_headline = self.word_embeds(headlines)  # [N, L_headline, H_embed]
        x_body = self.word_embeds(bodys)  # [N, P, L_para, H_embed]

        _, h_headline = self.headline_encoder(pack_padded_sequence(x_headline, headline_lengths, 
                                                                   batch_first=True, enforce_sorted=False))  # _, [1, N, H_enc]
        h_headline = h_headline.squeeze()  # [N, H_enc]

        valid_para_lengths = (para_lengths != 0).sum(dim=1).tolist()
        para_mask = (para_lengths == 0)
        # merge dimensions N and P
        para_lengths = para_lengths.flatten()
        para_lengths_masked = para_lengths[para_lengths != 0]
        x_paras_masked = x_body.flatten(0,1)[para_lengths != 0]

        _, h_paras_masked = self.paragraph_encoder(pack_padded_sequence(x_paras_masked, para_lengths_masked, 
                                                                        batch_first=True, enforce_sorted=False))

        # unmerge dimensions N and P
        h_paras_grouped = h_paras_masked.squeeze().split(valid_para_lengths)
        h_paras = pad_sequence(h_paras_grouped)

        output_body_packed, _ = self.body_encoder(pack_padded_sequence(h_paras, valid_para_lengths, enforce_sorted=False))
        output_body, _ = pad_packed_sequence(output_body_packed, 
                                             batch_first=True, total_length=para_mask.shape[-1])  # [N, P, 2 * H]

        h_body = self.attention(output_body, para_mask, h_headline)

        return self.bilinear(h_headline, h_body).squeeze()
