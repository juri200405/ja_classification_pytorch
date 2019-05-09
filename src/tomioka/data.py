import MeCab

m = MeCab.Tagger("-O wakati")
print(m.parse("本日は晴天なり"))