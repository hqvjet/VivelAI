import numpy as np
from vncorenlp import VnCoreNLP
import re

from exception_dict import token_dict

# INITIALIZE VAR
rdr = VnCoreNLP("tools/vncorenlp/VnCoreNLP-1.1.1.jar", annotators="wseg", max_heap_size='-Xmx500m')

def useTokenize(texts):
    res = []
    for text in texts:
        temp = []
        for arr_sentence in rdr.tokenize(text):
            sequence = [word.lower() for word in arr_sentence]
            tokened_sequence = filterRareToken(sequence)
            temp.append(' '.join(tokened_sequence))
        res.append(' '.join(temp))

    return np.array([starRating(sentence) for sentence in res])

def starRating(text):
    pattern = r"\b([0-5])\s*(stars?|sao|\*)?" 
    replacement = r"\1_\2"

    def replace(match):
        number = match.group(1)
        word = number_map.get(number, number)
        return f"{word}_{match.group(2)}"

    return re.sub(pattern, replacement, text, flags=re.IGNORECASE)

def filterRareToken(arr_text):
    final_token = []

    i = 0
    while i < len(arr_text) - 1:
        # print(arr_text[i] + ' ' + arr_text[i + 1], (arr_text[i] + ' ' + arr_text[i + 1]) in token_dict)
        if (arr_text[i] + ' ' + arr_text[i + 1]) in token_dict:
            final_token.append(token_dict[(arr_text[i] + ' ' + arr_text[i + 1])])
            i += 1
        else:
            final_token.append(arr_text[i])
        i += 1

    if arr_text[-2] + ' ' + arr_text[-1] not in token_dict:
        final_token.append(arr_text[-1])

    return final_token

s = useTokenize(['Tôi là Hoàng Quốc Việt, đang check in quầy', 'Tôi có 5sao, 1 star, và 3* đánh giá. Còn bạn thì sao?'])
print(s)
