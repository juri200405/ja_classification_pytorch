import MeCab
import pickle
from pathlib import Path
import csv
import re
import unicodedata
from collections import Counter


START_TOKEN = "<s>"
END_TOKEN = "</s>"
PAD_TOKEN = "<pad>"
UNK_TOKEN = "<unk>"

def normalize_string(s):
    s = ''.join(c for c in unicodedata.normalize('NFKC', s) if unicodedata.category(c) != 'Mn')
    s = re.sub(r'[\r\n]+', r'\n', s).strip()
    return s

def preprocess(me, sample):
    sentence_list = re.split(r'[。\n]+', normalize_string(sample["data"]))
    wakati = (' ' + END_TOKEN + ' ' + START_TOKEN + ' ').join(me.parse(s.strip()).strip() for s in sentence_list)
    tokens = [START_TOKEN]
    tokens += re.split(r' ', wakati)
    tokens += [END_TOKEN]
    return {"tokens": tokens, "label": int(sample["label"])}

def build_voc(tokens, max_size=50000):
    counter = Counter(tokens)
    words, _ = zip(*counter.most_common(max_size))
    words = [PAD_TOKEN, UNK_TOKEN] + list(words)
    token2index = {word: index for index, word in enumerate(words)}

    if START_TOKEN not in token2index:
        token2index[START_TOKEN] = len(token2index)
        words += [START_TOKEN]

    if END_TOKEN not in token2index:
        token2index[END_TOKEN] = len(token2index)
        words += [END_TOKEN]

    return words, token2index

def build(dpath, savedir):
    data = []
    with open(str(dpath / "train.csv"), 'rt', newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        me = MeCab.Tagger("-O wakati")
        data = [preprocess(me, row) for row in reader]
    words, tokens = build_voc(data[0]["tokens"])
    print(tokens)

if __name__ == "__main__":
    build(Path("../../datas"), Path("../../datasets"))