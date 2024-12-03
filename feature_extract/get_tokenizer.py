from transformers import AutoModel, AutoTokenizer
from constant import PHOBERT_VER, EMOJI_NEG, EMOJI_NEU, EMOJI_POS

def getTokenizer():
    tokenizer = AutoTokenizer.from_pretrained(PHOBERT_VER)
    toks = [EMOJI_NEG, EMOJI_NEU, EMOJI_POS]

    num_added = tokenizer.add_tokens(toks)

    return tokenizer
