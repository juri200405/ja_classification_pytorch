import MeCab
import pickle
from pathlib import Path
import csv
import re
import unicodedata


START_TOKEN = "<s>"
END_TOKEN = "</s>"
PAD_TOKEN = "<pad>"
UNK_TOKEN = "<unk>"

def normalize_string(s):
    s = ''.join(c for c in unicodedata.normalize('NFKC', s) if unicodedata.category(c) != 'Mn')
    s = re.sub(r'[\r\n]+', r'\n', s).strip()
    return s

def preprocess(me, sample):
    wakati = ' 。 '.join(me.parse(s.strip()).strip() for s in re.split(r'[。\n]+', normalize_string(sample["data"])))
    tokens = re.split(r' ', wakati )
    return {"tokens": tokens, "label": int(sample["label"])}

def build(dpath, savedir):
    data = []
    with open(str(dpath / "train.csv"), 'rt', newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        me = MeCab.Tagger("-O wakati")
        data = [preprocess(me, row) for row in reader]
    print(data[0])

if __name__ == "__main__":
    build(Path("../../datas"), Path("../../datasets"))