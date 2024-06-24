import re

from feature_extract.exception_dict import lemmatization_dict

def useLemma(texts):
    for i in range(len(texts)):
        arr_text = wordSegment(texts[i])
        texts[i] = filterLemmatization(arr_text)

    return texts

def wordSegment(text):
    pattern = r"\w+|[^\w\s]"
    words = re.findall(pattern, text)
    return words

def filterLemmatization(arr_text):
    final_text = []

    i = 0
    while i < len(arr_text) - 1:
        if (arr_text[i] + ' ' + arr_text[i + 1]) in lemmatization_dict:
            final_text.append(lemmatization_dict[(arr_text[i] + ' ' + arr_text[i + 1])])
            i += 1
        elif arr_text[i] in lemmatization_dict:
            final_text.append(lemmatization_dict[arr_text[i]])
        else:
            final_text.append(arr_text[i])
        i += 1

    if arr_text[-1] in lemmatization_dict:
        final_text.append(lemmatization_dict[arr_text[-1]])
    else:
        final_text.append(arr_text[-1])

    return ' '.join(final_text)
