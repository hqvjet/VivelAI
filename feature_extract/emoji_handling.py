import emoji
from constant import *
import json
import pandas as pd
import re
import torch
import torchtext.vocab as tvocab
import torch.nn as nn

word_embedding = tvocab.Vectors(name=f'res/emoji2vec.txt', unk_init=torch.Tensor.normal_)
with open('res/emoji_dict.json', 'r', encoding='utf-8') as f:
    emoji_dict = json.load(f)
emoji_dict = emoji_dict['dict']

def replace_emoji_with_unicode(text):
    def emoji_to_unicode(match):
        emoji_char = match.group(0)
        unicode_point = ''
        for c in emoji_char:
            unicode_point += f'0x{ord(c):x}'

        return unicode_point

    emoji_pattern = re.compile("[" + "".join(emoji.EMOJI_DATA.keys()) + "]")

    return emoji_pattern.sub(lambda m: f"{emoji_to_unicode(m)}", text)

def replace_emoji_with_text(text):
    def emoji_to_text(match):
        emoji = match.group(0)
        unicode = ''
        for c in emoji:
            unicode += f'0x{ord(c):x}'
        if unicode in emoji_dict:
            e_text = emoji_dict[unicode]
            if e_text is None:
                return ''
            return e_text
        else:
            return ''

    emoji_pattern = re.compile("[" + "".join(emoji.EMOJI_DATA.keys()) + "]")

    return emoji_pattern.sub(lambda m: f"{emoji_to_text(m)}", text)

def emojiHandling(texts, mode=None):
    MODE_LIST = ['unicode', 'text']
    if mode not in MODE_LIST:
        raise ValueError(f"Emoji handling mode must be one of {MODE_LIST}")

    returned = []
    if mode == MODE_LIST[0]:
        for i in range(len(texts)):
            returned.append(replace_emoji_with_unicode(texts[i])) 
    elif mode == MODE_LIST[1]:
        for i in range(len(texts)):
            returned.append(replace_emoji_with_text(texts[i]))

    return returned

def getEmojiEmbeddingMatrix():
    def convert_emoji2unicode(emoji):
        return f"0x{ord(emoji):x}"

    list_emoji = list(word_embedding.stoi.keys())
    e_matrix = {}
    linear = nn.Linear(300, 768)

    for emoji in list_emoji:
        if len(emoji) == 1:
            key = convert_emoji2unicode(emoji)
            val = word_embedding.vectors[word_embedding.stoi[emoji]]
            val = torch.tensor(val, dtype=torch.float)
            emb = linear(val)
            e_matrix[key] = emb

    return e_matrix
