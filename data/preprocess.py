#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# @Version : Python 3.6

import json
import re
from tqdm import tqdm
from stanfordcorenlp import StanfordCoreNLP
nlp = StanfordCoreNLP(r'./stanford-corenlp-full-2018-10-05')
props = dict(
    annotators='tokenize,pos',
    pipelineLanguage='en',
    outputFormat='json',
)


def sentence_process(raw_sentence):
    sentence = raw_sentence[1:-1]  # remove quotes
    res = json.loads(nlp.annotate(sentence, properties=props))
    sents = res['sentences']
    token = [x['word'] for sent in sents for x in sent['tokens']]
    pos = [x['pos'] for sent in sents for x in sent['tokens']]

    lengths = map(len, [token, pos])
    assert len(set(lengths)) == 1
    assert '<e1>' in token
    assert '<e2>' in token
    assert '</e1>' in token
    assert '</e2>' in token
    return remove_postion_indicators(token, pos)


def remove_postion_indicators(token, pos):
    subj_start = subj_end = obj_start = obj_end = 0
    pure_token = []
    pure_pos = []
    for i, word in enumerate(token):
        if '<e1>' == word:
            subj_start = len(pure_token)
            continue
        if '</e1>' == word:
            subj_end = len(pure_token) - 1
            continue
        if '<e2>' in word:
            obj_start = len(pure_token)
            continue
        if '</e2>' in word:
            obj_end = len(pure_token) - 1
            continue
        pure_token.append(word)
        pure_pos.append(pos[i])
    res = dict(
        token=pure_token,
        subj_start=subj_start,
        subj_end=subj_end,
        obj_start=obj_start,
        obj_end=obj_end,
        stanford_pos=pure_pos
    )
    return res


def convert(src_file, des_file):
    with open(src_file, 'r', encoding='utf-8') as fr:
        file_data = fr.readlines()

    with open(des_file, 'w', encoding='utf-8') as fw:
        for i in tqdm(range(0, len(file_data), 4)):
            meta = {}
            s = file_data[i].strip().split('\t')
            assert len(s) == 2
            meta['id'] = s[0]
            meta['relation'] = file_data[i+1].strip()
            meta['comment'] = file_data[i+2].strip()
            sen_res = sentence_process(s[1])
            json.dump({**meta, **sen_res}, fw, ensure_ascii=False)
            fw.write('\n')


if __name__ == '__main__':
    path_train = './SemEval2010_task8_all_data/SemEval2010_task8_training/TRAIN_FILE.TXT'
    path_test = './SemEval2010_task8_all_data/SemEval2010_task8_testing_keys/TEST_FILE_FULL.TXT'

    convert(path_train, 'train.json')
    convert(path_test, 'test.json')

    nlp.close()
