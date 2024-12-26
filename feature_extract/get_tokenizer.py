from transformers import AutoModel, AutoTokenizer
from constant import PHOBERT_VER, EMOJI_NEG, EMOJI_NEU, EMOJI_POS

def getTokenizer(e_matrix):
    tokenizer = AutoTokenizer.from_pretrained(PHOBERT_VER)
    # toks = [EMOJI_NEG, EMOJI_NEU, EMOJI_POS]
    toks = list(e_matrix.keys())

    num_added = tokenizer.add_tokens(toks)

    return tokenizer
