from transformers import AutoModel, AutoTokenizer
from constant import *

def getTokenizer(extract_model, e_matrix=None):
    tokenizer = AutoTokenizer.from_pretrained(
        PHOBERT_VER if extract_model != VISOBERT else VISOBERT_VER, 
        # use_fast=False if extract_model == VISOBERT else True,
        force_download=True
    )

    if e_matrix != None:
        toks = list(e_matrix.keys())
        num_added = tokenizer.add_tokens(toks)

    return tokenizer
