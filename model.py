#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# @Version : Python 3.6

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init


class Attention_CNN(nn.Module):
    def __init__(self, word_vec, class_num, pos_num, config):
        super().__init__()
        self.word_vec = word_vec
        self.class_num = class_num
        self.pos_num = pos_num

        # hyper parameters and others
        self.max_len = config.max_len
        self.word_dim = config.word_dim
        self.pos_dim = config.pos_dim
        self.pos_dis = config.pos_dis
        self.tag_dim = config.tag_dim

        self.dropout_value = config.dropout
        self.filter_num = config.filter_num
        self.window = config.window

        self.dim = self.word_dim + 2 * self.pos_dim + self.tag_dim

        # net structures and operations
        self.word_embedding = nn.Embedding.from_pretrained(
            embeddings=self.word_vec,
            freeze=False,
        )
        self.pos1_embedding = nn.Embedding(
            num_embeddings=2 * self.pos_dis + 3,
            embedding_dim=self.pos_dim
        )
        self.pos2_embedding = nn.Embedding(
            num_embeddings=2 * self.pos_dis + 3,
            embedding_dim=self.pos_dim
        )

        self.tag_embedding = nn.Embedding(
            num_embeddings=self.pos_num,
            embedding_dim=self.tag_dim
        )
        self.conv = nn.Conv2d(
            in_channels=1,
            out_channels=self.filter_num,
            kernel_size=(self.window, self.dim),
            stride=(1, 1),
            bias=True,
            padding=(1, 0),  # same padding
            padding_mode='zeros'
        )
        self.maxpool = nn.MaxPool2d((self.max_len, 1))
        self.tanh = nn.Tanh()
        self.we = nn.Linear(
            in_features=self.dim * 2,
            out_features=self.dim * 2,
            bias=True
        )
        self.wa = nn.Linear(
            in_features=self.dim*2,
            out_features=1,
            bias=True
        )
        self.dense = nn.Linear(
            in_features=self.filter_num + 2 * self.dim,
            out_features=self.class_num,
            bias=True
        )

        # initialize weight
        init.uniform_(self.pos1_embedding.weight, a=-0.1, b=0.1)
        init.uniform_(self.pos2_embedding.weight, a=-0.1, b=0.1)
        init.uniform_(self.tag_embedding.weight, a=-0.1, b=0.1)
        init.uniform_(self.conv.weight, a=-0.1, b=0.1)
        init.constant_(self.conv.bias, 0.)
        init.uniform_(self.we.weight, a=-0.1, b=0.1)
        init.constant_(self.we.bias, 0.)
        init.uniform_(self.wa.weight, a=-0.1, b=0.1)
        init.constant_(self.wa.bias, 0.)
        init.uniform_(self.dense.weight, a=-0.1, b=0.1)
        init.constant_(self.dense.bias, 0.)

    def encoder_layer(self, token, pos1, pos2, tags):
        word_emb = self.word_embedding(token)  # B*L*word_dim
        pos1_emb = self.pos1_embedding(pos1)  # B*L*pos_dim
        pos2_emb = self.pos2_embedding(pos2)  # B*L*pos_dim
        tag_emb = self.tag_embedding(tags)  # B*L*tag_dim
        emb = torch.cat(tensors=[word_emb, pos1_emb, pos2_emb, tag_emb], dim=-1)
        return emb  # B*L*D, D=word_dim+2*pos_dim+tag_dim

    def conv_layer(self, emb, mask):
        emb = emb.unsqueeze(dim=1)  # B*1*L*D
        conv = self.conv(emb)  # B*C*L*1

        # mask, remove the effect of 'PAD'
        conv = conv.view(-1, self.filter_num, self.max_len)  # B*C*L
        mask = mask.unsqueeze(dim=1)  # B*1*L
        mask = mask.expand(-1, self.filter_num, -1)  # B*C*L
        conv = conv.masked_fill_(mask.eq(0), float('-inf'))  # B*C*L
        conv = conv.unsqueeze(dim=-1)  # B*C*L*1
        return conv

    def single_maxpool_layer(self, conv):
        pool = self.maxpool(conv)  # B*C*1*1
        pool = pool.view(-1, self.filter_num)  # B*C
        return pool

    def entity_average(self, emb, e_mask):
        lengths = torch.sum(e_mask.eq(1), dim=-1).view(-1, 1)  # B*1
        mask = e_mask.unsqueeze(dim=1).float()  # B*1*L
        sum_emb = torch.bmm(mask, emb).squeeze(dim=1)  # B*D
        avg_emb = sum_emb / lengths  # B*D, broadcasting
        return avg_emb

    def attention_layer(self, emb, entity, mask):
        entity = entity.unsqueeze(dim=1).expand(-1, self.max_len, -1)  # B*L*D
        h = torch.cat(tensors=[emb, entity], dim=-1)  # B*L*2D
        h_flat = h.view(-1, 2*self.dim)  # B·L*2D
        output = self.tanh(self.we(h_flat))  # B·L*2D
        u_flat = self.wa(output)  # B·L*1
        u = u_flat.view(-1, self.max_len)  # B*L

        # remove the effect of <PAD>
        att_score = u.masked_fill(mask.eq(0), float('-inf'))  # B*L
        att_weight = F.softmax(att_score, dim=-1).unsqueeze(dim=-1)  # B*L*1

        reps = torch.bmm(emb.transpose(1, 2), att_weight).squeeze(dim=-1)  # B*D*L * B*L*1 -> B*D*1 -> B*D
        return reps

    def forward(self, data):
        token = data[:, 0, :].view(-1, self.max_len)
        pos1 = data[:, 1, :].view(-1, self.max_len)
        pos2 = data[:, 2, :].view(-1, self.max_len)
        mask = data[:, 3, :].view(-1, self.max_len)
        tags = data[:, 4, :].view(-1, self.max_len)
        e1_mask = data[:, 5, :].view(-1, self.max_len)
        e2_mask = data[:, 6, :].view(-1, self.max_len)
        emb = self.encoder_layer(token, pos1, pos2, tags)

        conv = self.conv_layer(emb, mask)
        conv = self.tanh(conv)
        pool = self.single_maxpool_layer(conv)

        e1_emb = self.entity_average(emb, e1_mask)
        e2_emb = self.entity_average(emb, e2_mask)
        e1_context = self.attention_layer(emb, e1_emb, mask)
        e2_context = self.attention_layer(emb, e2_emb, mask)
        e1_context = self.tanh(e1_context)
        e2_context = self.tanh(e2_context)

        feature = torch.cat(tensors=[pool, e1_context, e2_context], dim=-1)  # B* C+2D
        logits = self.dense(feature)
        return logits
