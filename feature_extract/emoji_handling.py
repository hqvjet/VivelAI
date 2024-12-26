import emoji
from constant import EMOJI_NEG, EMOJI_NEU, EMOJI_POS
import pandas as pd
import re
import torch
import torch.nn as nn

emoji_data = pd.read_csv('res/emoji_sentiment.csv')

# def convert_emoji2text(text):
#     emoji_list = emoji.emoji_list(text)
#     
#     unicode_emoji = []
#     for e in emoji_list:
#         if len(e['emoji']) == 1:
#             unicode_emoji.append(f"0x{ord(e['emoji']):x}")
#         else:
#             unicode_emoji.append('error')
#     
#     for e, unicode_val in zip(emoji_list, unicode_emoji):
#         emoji_sentiment = emoji_data[emoji_data['Unicode codepoint'] == unicode_val]
#         if emoji_sentiment.empty:
#             emoji_sentiment = EMOJI_NEG
#         else:
#             neg = emoji_sentiment['Negative'].values[0]
#             neu = emoji_sentiment['Neutral'].values[0]
#             pos = emoji_sentiment['Positive'].values[0]
#
#             if neg > neu and neg > pos:
#                 emoji_sentiment = ' ' + EMOJI_NEG + ' '
#             elif neu > pos:
#                 emoji_sentiment = ' ' + EMOJI_NEU + ' '
#             else:
#                 emoji_sentiment = ' ' + EMOJI_POS + ' '
#
#         text = text.replace(e['emoji'], emoji_sentiment)
#     
#     return text

def replace_emoji_with_unicode(text):
    def emoji_to_unicode(match):
        emoji_char = match.group(0)
        unicode_point = f" 0x{ord(emoji_char):x} "
        return unicode_point

    emoji_pattern = re.compile("[" + "".join(emoji.EMOJI_DATA.keys()) + "]")

    return emoji_pattern.sub(lambda m: f"{emoji_to_unicode(m)}", text)

def emojiHandling(texts):
    returned = []
    for i in range(len(texts)):
        returned.append(replace_emoji_with_unicode(texts[i])) 

    return returned

def getEmojiEmbeddingMatrix():
    emojis = emoji_data.values
    e_matrix = {}
    linear = nn.Linear(1, 768)

    for emoji in emojis:
        occu = emoji[3]
        for status in emoji[4:7]:
            emb = []
            if status != 0:
                emb.append(occu / status)
            else:
                emb.append(0)

        emb = torch.tensor(emb, dtype=torch.float)
        emb = emb.unsqueeze(0)
        emb = linear(emb)
        e_matrix[emoji[1]] = emb

    return e_matrix
