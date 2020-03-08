#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# @Version : Python 3.6

from gensim.models import KeyedVectors

model = KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True)
model.save_word2vec_format('GoogleNews-vectors-negative300.txt', binary=False)
