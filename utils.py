#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# @Version : Python 3.6


import os
import json
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader


class WordEmbeddingLoader(object):
    """
    A loader for pre-trained word embedding
    """

    def __init__(self, config):
        self.embedding_path = config.embedding_path  # path of pre-trained word embedding
        self.embedding_dim = config.word_dim  # dimension of word embedding
        self.data_dir = config.data_dir
        self.min_freq = config.min_freq
        self.cache_dir = config.cache_dir

    def __build_vocab(self):
        vocab = {}
        filename = ['train.json', 'test.json']
        # filename = ['train.json']
        for fn in filename:
            print('building vocaburary from %s' % fn)
            with open(os.path.join(self.data_dir, fn), 'r', encoding='utf-8') as fr:
                for line in fr:
                    line = json.loads(line.strip())
                    sentence = line['token']
                    for token in sentence:
                        token = token.lower()
                        vocab[token] = vocab.get(token, 0) + 1
        vocab = set([token for token in vocab if vocab[token] > self.min_freq])
        return vocab

    def __load_embedding(self):
        vocab = self.__build_vocab()
        token2id = {}
        token2id['PAD'] = len(token2id)
        token2id['UNK'] = len(token2id)
        token_emb = []
        with open(self.embedding_path, 'r', encoding='utf-8') as fr:
            for line in fr:
                line = line.strip().split()
                if len(line) != self.embedding_dim + 1:
                    continue
                if line[0] not in vocab:
                    continue
                token2id[line[0]] = len(token2id)
                token_emb.append(np.asarray(line[1:], dtype=np.float32))
        token_emb = np.stack(token_emb).reshape(-1, self.embedding_dim)
        special_emb = np.random.uniform(-0.1, 0.1, size=(2, self.embedding_dim))
        token_emb = np.concatenate((special_emb, token_emb), axis=0)

        token_emb = token_emb.astype(np.float32).reshape(-1, self.embedding_dim)
        token_emb = torch.from_numpy(token_emb)
        return token2id, token_emb

    def load_embedding(self):
        data_cache = os.path.join(self.cache_dir, 'GoogleNews.pkl')
        if not os.path.exists(data_cache):
            token2id, token_emb = self.__load_embedding()
            torch.save([token2id, token_emb], data_cache)
        else:
            token2id, token_emb = torch.load(data_cache)
        print('embedding scale: {}*{}d'.format(len(token2id), self.embedding_dim))
        print('finish loading embeddng!')
        return token2id, token_emb


class RelationLoader(object):
    def __init__(self, config):
        self.data_dir = config.data_dir

    def __load_relation(self):
        relation_file = os.path.join(self.data_dir, 'relation2id.txt')
        rel2id = {}
        id2rel = {}
        with open(relation_file, 'r', encoding='utf-8') as fr:
            for line in fr:
                relation, id_s = line.strip().split()
                id_d = int(id_s)
                rel2id[relation] = id_d
                id2rel[id_d] = relation
        return rel2id, id2rel, len(rel2id)

    def get_relation(self):
        return self.__load_relation()


"""
POS_MAP is a mapping dict for POS tags that customized by myself,
and it may not be right.
Section 3.1.1 `Part-of-speech tag Embeddings` mentioned a coarse-grained
POS category, containing 15 different tags.
"""
POS_MAP = {
    'NN': 'noun',
    'NNS': 'noun',
    'NNP': 'noun',
    'NNPS': 'noun',
    'DT': 'determiner',
    'PDT': 'determiner',
    'IN': 'preposition',
    'RP': 'preposition',
    'TO': 'preposition',
    'JJ': 'adjective',
    'JJR': 'adjective',
    'JJS': 'adjective',
    '.': 'punctuation',
    ',': 'punctuation',
    '-RRB-': 'punctuation',
    '-LRB-': 'punctuation',
    '\'\'': 'punctuation',
    '``': 'punctuation',
    ':': 'punctuation',
    '$': 'punctuation',
    'SYM': 'punctuation',
    '#': 'punctuation',
    'VBN': 'verb',
    'VBD': 'verb',
    'VBZ': 'verb',
    'VBG': 'verb',
    'VBP': 'verb',
    'VB': 'verb',
    'CC': 'conjunction',
    'RB': 'adverb',
    'RBR': 'adverb',
    'RBS': 'adverb',
    'CD': 'numeral',
    'LS': 'numeral',
    'PRP': 'pronoun',
    'PRP$': 'pronoun',
    'POS': 'pronoun',
    'WDT': 'wh-',
    'WRB': 'wh-',
    'WP': 'wh-',
    'WP$': 'wh-',
    'MD': 'modal-auxiliary',
    'EX': 'existential',
    'UH': 'uh',
    'FW': 'foreign-word',
}


class PosLoader(object):
    def __init__(self, config):
        self.data_dir = config.data_dir

    def __load_pos(self):
        tags = {}
        with open(os.path.join(self.data_dir, 'train.json'), 'r', encoding='utf-8') as fr:
            for line in fr:
                line = json.loads(line.strip())
                pos = line['stanford_pos']
                for tag in pos:
                    tag = POS_MAP[tag]
                    tags[tag] = tags.get(tag, 0) + 1
        with open(os.path.join(self.data_dir, 'test.json'), 'r', encoding='utf-8') as fr:
            for line in fr:
                line = json.loads(line.strip())
                pos = line['stanford_pos']
                for tag in pos:
                    tag = POS_MAP[tag]
                    tags[tag] = tags.get(tag, 0) + 1
        tags = sorted(tags.items(), key=lambda x: x[1], reverse=True)
        pos2id = {tag[0]: i for i, tag in enumerate(tags)}
        pos2id['PAD'] = len(pos2id)
        # print(pos2id)
        return pos2id

    def get_pos(self):
        return self.__load_pos()


class SemEvalDateset(Dataset):
    def __init__(self, filename, rel2id, pos2id, word2id, config):
        self.filename = filename
        self.rel2id = rel2id
        self.pos2id = pos2id
        self.word2id = word2id
        self.max_len = config.max_len
        self.pos_dis = config.pos_dis
        self.data_dir = config.data_dir
        self.dataset, self.label = self.__load_data()

    def __get_pos_index(self, x):
        if x < -self.pos_dis:
            return 0
        if x >= -self.pos_dis and x <= self.pos_dis:
            return x + self.pos_dis + 1
        if x > self.pos_dis:
            return 2 * self.pos_dis + 2

    def __get_relative_pos(self, x, entity_pos):
        if x < entity_pos[0]:
            return self.__get_pos_index(x-entity_pos[0])
        elif x > entity_pos[1]:
            return self.__get_pos_index(x-entity_pos[1])
        else:
            return self.__get_pos_index(0)

    def __symbolize_sentence(self, e1_pos, e2_pos, sentence, pos):
        """
            Args:
                e1_pos (tuple) span of e1
                e2_pos (tuple) span of e2
                sentence (list)
                pos (list) POS of words
        """
        mask = [1] * len(sentence)
        if e1_pos[0] < e2_pos[0]:
            for i in range(e1_pos[0], e2_pos[1]+1):
                mask[i] = 2
            for i in range(e2_pos[1]+1, len(sentence)):
                mask[i] = 3
        else:
            for i in range(e2_pos[0], e1_pos[1]+1):
                mask[i] = 2
            for i in range(e1_pos[1]+1, len(sentence)):
                mask[i] = 3

        e1_mask = [0] * self.max_len
        e2_mask = [0] * self.max_len
        for i in range(e1_pos[0], e1_pos[1]+1):
            e1_mask[i] = 1
        for i in range(e2_pos[0], e2_pos[1]+1):
            e2_mask[i] = 1

        words = []
        pos1 = []
        pos2 = []
        tags = []
        length = min(self.max_len, len(sentence))
        mask = mask[:length]

        for i in range(length):
            words.append(self.word2id.get(sentence[i].lower(), self.word2id['UNK']))
            pos1.append(self.__get_relative_pos(i, e1_pos))
            pos2.append(self.__get_relative_pos(i, e2_pos))
            tags.append(self.pos2id[POS_MAP[pos[i]]])

        if length < self.max_len:
            for i in range(length, self.max_len):
                mask.append(0)  # 'PAD' mask is zero
                words.append(self.word2id['PAD'])
                tags.append(self.pos2id['PAD'])

                pos1.append(self.__get_relative_pos(i, e1_pos))
                pos2.append(self.__get_relative_pos(i, e2_pos))
        unit = np.asarray([words, pos1, pos2, mask, tags, e1_mask, e2_mask], dtype=np.int64)
        unit = np.reshape(unit, newshape=(1, 7, self.max_len))
        return unit

    def __load_data(self):
        path_data_file = os.path.join(self.data_dir, self.filename)
        data = []
        labels = []
        with open(path_data_file, 'r', encoding='utf-8') as fr:
            for line in fr:
                line = json.loads(line.strip())
                label = line['relation']
                sentence = line['token']
                pos = line['stanford_pos']
                e1_pos = (line['subj_start'], line['subj_end'])
                e2_pos = (line['obj_start'], line['obj_end'])
                label_idx = self.rel2id[label]

                one_sentence = self.__symbolize_sentence(e1_pos, e2_pos, sentence, pos)
                data.append(one_sentence)
                labels.append(label_idx)
        return data, labels

    def __getitem__(self, index):
        data = self.dataset[index]
        label = self.label[index]
        return data, label

    def __len__(self):
        return len(self.label)


class SemEvalDataLoader(object):
    def __init__(self, rel2id, pos2id, word2id, config):
        self.rel2id = rel2id
        self.pos2id = pos2id
        self.word2id = word2id
        self.config = config

    def __collate_fn(self, batch):
        data, label = zip(*batch)  # unzip the batch data
        data = list(data)
        label = list(label)
        data = torch.from_numpy(np.concatenate(data, axis=0))
        label = torch.from_numpy(np.asarray(label, dtype=np.int64))
        return data, label

    def __get_data(self, filename, shuffle=False):
        dataset = SemEvalDateset(filename, self.rel2id, self.pos2id, self.word2id, self.config)
        loader = DataLoader(
            dataset=dataset,
            batch_size=self.config.batch_size,
            shuffle=shuffle,
            num_workers=2,
            collate_fn=self.__collate_fn
        )
        return loader

    def get_train(self):
        return self.__get_data('train.json', shuffle=True)

    def get_dev(self):
        return self.__get_data('test.json', shuffle=False)

    def get_test(self):
        return self.__get_data('test.json', shuffle=False)


if __name__ == '__main__':
    from config import Config
    config = Config()
    word2id, word_vec = WordEmbeddingLoader(config).load_embedding()
    rel2id, id2rel, class_num = RelationLoader(config).get_relation()
    pos2id = PosLoader(config).get_pos()

    loader = SemEvalDataLoader(rel2id, pos2id, word2id, config)
    test_loader = loader.get_train()

    for step, (data, label) in enumerate(test_loader):
        print(type(data), data.shape)
        print(type(label), label.shape)
        break
