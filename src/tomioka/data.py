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
    '''
    文s中に含まれる文字を正規化し、連続する改行を一つの改行に置き換える。
    '''
    s = ''.join(c for c in unicodedata.normalize('NFKC', s) if unicodedata.category(c) != 'Mn')
    s = re.sub(r'[\r\n]+', r'\n', s).strip()
    return s

def preprocess(sample):
    '''
    入力：{"data": <文章>, "label": <ラベル>}
    出力：{"tokens": <単語の羅列>, "label": <ラベル>}

    入力の文章を正規化して分かち書き。それを単語の羅列に変換する。
    '''
    me = MeCab.Tagger("-O wakati")
    sentence_list = re.split(r'[。\n]+', normalize_string(sample["data"]))
    wakati = (' ' + END_TOKEN + ' ' + START_TOKEN + ' ').join(me.parse(s.strip()).strip() for s in sentence_list)
    tokens = [START_TOKEN]
    tokens += re.split(r' ', wakati)
    tokens += [END_TOKEN]
    if sample["label"] == '':
        return {"tokens": tokens, "label": None}
    else:
        return {"tokens": tokens, "label": int(sample["label"])}

def build_voc(tokens, max_size=50000):
    '''
    入力された単語の羅列を、「単語のリスト」と「単語とインデックスの対応表」に変換
    '''
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

def postprocess(t2i, unk_index, sample):
    '''
    単語の羅列を、インデックスの羅列に変換
    '''
    return {'tokens': [t2i.get(token, unk_index) for token in sample['tokens']], 'label': sample['label']}

def build_helper(dpath, filename):
    '''
    csvファイルから、その内容を取得
    '''
    data = []
    with open(str(dpath / filename), 'rt', newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        data = [preprocess(row) for row in reader]
    return data

def build(dpath, savedir):
    '''
    dpathにあるcsvファイルから、学習データを生成
    '''
    train_data = build_helper(dpath, "train.csv")
    test_data = build_helper(dpath, "test.csv")
    
    all_token = []
    for d in train_data:
        all_token += d["tokens"]
    words, token2index = build_voc(all_token)
    
    with open(savedir / 'vocab.pkl', 'wb') as f:
        pickle.dump((token2index, words), f)

    train_data_index = [postprocess(token2index, token2index[UNK_TOKEN], item) for item in train_data]
    with open(savedir / 'dataset.train.token.pkl', 'wb') as f:
        pickle.dump(train_data_index, f)
    
    test_data_index = [postprocess(token2index, token2index[UNK_TOKEN], item) for item in test_data]
    with open(savedir / 'dataset.test.token.pkl', 'wb') as f:
        pickle.dump(test_data_index, f)

if __name__ == "__main__":
    build(Path("./datas"), Path("./datas/datasets"))