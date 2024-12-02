import emoji
from constant import EMOJI_NEG, EMOJI_NEU, EMOJI_POS
import pandas as pd

emoji_data = pd.read_csv('res/emoji_sentiment.csv')

def convert_emoji2text(text):
    emoji_list = emoji.emoji_list(text)
    
    unicode_emoji = []
    for e in emoji_list:
        if len(e['emoji']) == 1:
            unicode_emoji.append(f"0x{ord(e['emoji']):x}")
        else:
            unicode_emoji.append('error')
    
    for e, unicode_val in zip(emoji_list, unicode_emoji):
        emoji_sentiment = emoji_data[emoji_data['Unicode codepoint'] == unicode_val]
        if emoji_sentiment.empty:
            emoji_sentiment = EMOJI_NEG
        else:
            neg = emoji_sentiment['Negative'].values[0]
            neu = emoji_sentiment['Neutral'].values[0]
            pos = emoji_sentiment['Positive'].values[0]

            if neg > neu and neg > pos:
                emoji_sentiment = ' ' + EMOJI_NEG + ' '
            elif neu > pos:
                emoji_sentiment = ' ' + EMOJI_NEU + ' '
            else:
                emoji_sentiment = ' ' + EMOJI_POS + ' '

        text = text.replace(e['emoji'], emoji_sentiment)
    
    return text

def emojiHandling(texts):
    returned = []
    for i in range(len(texts)):
        returned.append(convert_emoji2text(texts[i])) 

    return returned
