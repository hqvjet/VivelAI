from transformers import MBartForConditionalGeneration, MBart50TokenizerFast
import pandas as pd
import numpy as np

data = pd.read_csv('res/neg_neu.csv')

output = {
    'title': data['Title'].tolist(),
    'content': data['comment'].tolist(),
    'rating': data['rate'].tolist(),
    'label': data['label'].tolist()
}

# Load tokenizer và model mBART
tokenizer = MBart50TokenizerFast.from_pretrained("facebook/mbart-large-50-many-to-many-mmt")
model = MBartForConditionalGeneration.from_pretrained("facebook/mbart-large-50-many-to-many-mmt")

# Xác định ngôn ngữ đầu vào và đầu ra
tokenizer.src_lang = "vi_VN"  # Input là tiếng Việt
target_lang = "vi_VN"         # Output cũng là tiếng Việt

for i in range(len(output['content'])):
    if output['title'][i] is np.nan:
        continue

    if output['content'][i] == '' or output['content'][i] == np.nan or not isinstance(output['content'][i], str):
        continue

    inputs = tokenizer(output['content'][i], return_tensors="pt", padding=True, truncation=True)

    forced_bos_token_id = tokenizer.lang_code_to_id[target_lang]

    summary_ids = model.generate(inputs["input_ids"], num_beams=30, max_length=30, early_stopping=True, forced_bos_token_id=forced_bos_token_id)

    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    print(summary)
    output['title'][i] = summary

df = pd.DataFrame(output)
df.to_csv('res/full_neg_neu.csv', index=False)
print('Done')
