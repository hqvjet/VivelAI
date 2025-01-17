import pandas as pd
import random
import re

# Các nhóm biểu tượng
icons_mapping = {
    "pos": [  # Positive
        "😀", "😃", "😄", "😁", "😆", "😅", "😂", "🤣", "😊", "😇",
        "🙂", "😉", "😌", "😍", "🥰", "😘", "😗", "😙", "😚", "🤗",
        "🤩", "🤭", "🤠", "🥳", "💖", "💓", "💞", "💕", "💗", "💘",
        "💝", "🌟", "✨", "💫", "🎉", "🎊", "👏", "🙌", "👍", "💪",
        "🆗", "💖", "❤️", "🧡", "💛", "💚", "💙", "💜", "🖤", "🤍"
    ],
    "neu": [  # Neutral
        "😐", "😑", "😶", "🙃", "🧐", "🤨", "😏", "😒", "😬", "🤔",
        "🤷", "😕", "😟", "🤝", "👌", "✌️",
        "🤞", "🤙", "💆", "🙆",
        "👐", "🙌", "🤲", "🤝", "👋", "🤚", "🖖", "✋", "🤏"
    ],
    "neg": [  # Negative
        "😞", "😠", "😡", "🤬", "😭", "😢", "😿", "🙀", "💔", "😔",
        "😖", "😣", "😤", "😩", "😫", "🥵", "🥶", "🤒", "🤕", "🤧",
        "🥴", "😵", "🤯", "😰", "😨", "😧", "😦", "😬", "😿",
        "🙄", "💀", "☠️", "👿", "😈", "😒", "😓", "😑", "😞", "💢",
        "🤡", "👎", "🙅", "🚫", "❌", "🛑", "🤦"
    ]
}

# Hàm chọn biểu tượng dựa trên nhãn
def get_icons_by_rating(rating, num_icons):
    if rating == 0:
        r = 'neg'
    elif num_icons == 3 and rating == 1:
        r = 'neu'
    else:
        r = 'pos'

    if r in icons_mapping:
        icons = icons_mapping[r]
    else:
        icons = []  # Nếu không xác định được nhãn, không thêm biểu tượng
    return random.sample(icons, min(num_icons, len(icons)))

# Hàm phân phối biểu tượng vào giữa và cuối câu
def distribute_icons(text, icons):
    if not isinstance(text, str) or not text.strip():
        return text  # Nếu giá trị không phải chuỗi, trả lại như cũ
    sentences = re.split(r'(?<=[.!?]) +', text)  # Tách câu dựa trên dấu kết thúc câu

    # Thêm biểu tượng vào giữa hoặc cuối câu
    for i, sentence in enumerate(sentences):
        if i < len(icons):  # Đảm bảo không vượt quá số lượng biểu tượng
            sentences[i] += f" {icons[i]}"
    return " ".join(sentences)

# Đọc dữ liệu từ file Excel
input_files = ['AIVIVN_train', 'AIVIVN_test', 'UIT_VSFC_train', 'UIT_VSFC_test', 'UIT_ViHSD_train', 'UIT_ViHSD_test']  # Đường dẫn file của bạn
two_label_file = ['AIVIVN_train', 'AIVIVN_test']
for input_file in input_files:
    data = pd.read_csv(f'res/{input_file}.csv')

    if 'comment' not in data.columns or 'label' not in data.columns:
        raise KeyError("Cột 'content' hoặc 'rating' không tồn tại trong file dữ liệu!")

    data['comment'] = data.apply(
        lambda row: distribute_icons(row['comment'], get_icons_by_rating(row['label'], num_icons=2 if input_file in two_label_file else 3)), axis=1
    )

    output_file = f'res/{input_file}_emoji.csv'
    data.to_csv(output_file, index=False)

    print(f"Kết quả đã được lưu vào file: {output_file}")

