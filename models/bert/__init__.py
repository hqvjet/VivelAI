import math

from models.bert.positional_encoding import PositionalEncodingLayer
from models.bert.mha import MultiHeadAttention
from models.bert.config import layer_num, batch_size

def trainBERT(device, title, content, debug=False):
    batch_num = math.ceil(title.size(0) / batch_size)

    for i in range(batch_num):
        start = i * batch_size
        end = min(start + batch_size, title.size(0))

        batch_title = title[start:end]
        batch_content = content[start:end]

        batch_title = batch_title.to(device)
        batch_content = batch_content.to(device)

        pos_encoder = PositionalEncodingLayer(d_model=batch_title.size(2), max_len=batch_title.size(1), device=device)
        batch_title = pos_encoder(batch_title)
        batch_content = pos_encoder(batch_content)

        if debug:
            print(f'Shape of title and content after PE stage: {batch_title.size()}, {batch_content.size()}')
        
        for j in range(1):
            mha = MultiHeadAttention(device=device, head_num=15, d_model=batch_title.size(2))

            batch_title = mha(batch_title)
            batch_content = mha(batch_content)

            print(batch_title.size())

            # Multi-head attention: PE

            # Add & Norm: Concat(PE, MHA)

            # FeedForward: Add & Norm

            # Add & Norm: Concat(Previous Add & Norm, FeedForward)

        # Classification Layer: Final Add & Norm

        print(f'Batch number {i + 1} done!')
