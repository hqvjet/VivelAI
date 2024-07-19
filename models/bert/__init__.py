from models.bert.positional_encoding import PositionalEncodingLayer
from models.bert.config import *

def trainBERT(device, title, content):
    title = title.to(device)
    content = content.to(device)

    pos_encoder = PositionalEncodingLayer(d_model=title.size(2), max_len=title.size(1), device=device)
    title = pos_encoder(title)
    content = pos_encoder(content)

    for i in range(layer_num):
        pass           
        # Multi-head attention: PE

        # Add & Norm: Concat(PE, MHA)

        # FeedForward: Add & Norm

        # Add & Norm: Concat(Previous Add & Norm, FeedForward)

    # Classification Layer: Final Add & Norm
