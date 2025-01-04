import pandas as pd
import random
import re

# Các nhóm biểu tượng
icons_mapping = {
    "pos": ["😍", "😘", "😄", "😊", "😜","👏", "🌟", "💖", "😂", "❤","♥","🥰", "😎", "👌", "💕", "😁", "☺"
                "♡", "👍", "🙏", "✌", "😉", "😋", "💪", "😌", "😆", "😅", "😛", "😙", "⭐"],
    "neu": ["🙌", "😇", "🤔", "😶", "🙈", "😄", "😃"],
    "neg": ["😭", "😢", "😱", "😡", "😔", "😩", "😞", "😖", "😩", "😒", "😴", "😕", "😤", "😑", "😰"
                "😓", "😣", "😐", "😨", "👎", ],
    "food": [
        "🍎", "🍉", "🍇", "🍓", "🍒", "🍍", "🥭", "🥝", "🍔", "🍟", "🍕", "🌭",
        "🥪", "🌮", "🌯", "🍣", "🍤", "🍩", "🍰", "🎂", "🍨", "🍧", "🍦", "☕",
        "🍹", "🥤", "🍷", "🍺", "🍻", "🥂", "🥃"
    ],
    "transport": [
        "✈", "🚗", "🚕", "🚙", "🚌", "🚎", "🏎", "🚓", "🚑", "🚒", "🚐", "🚚",
        "🚜", "🚲", "🛵", "🏍", "🚤", "⛵", "🚢", "🚁", "🚂", "🚆", "🚊", "🚉",
        "🚀", "🛸", "🛳"
    ],
    "nature": [
        "🌸", "🌼", "🌻", "🌺", "🌷", "🌹", "🌲", "🌳", "🌴", "🌵", "🌾", "🌱",
        "☀", "🌤", "⛅", "🌥", "☁", "🌧", "⛈", "🌩", "🌪", "🌈", "❄", "🌊", "🔥"
    ],
    "activity": [
        "🎉", "🎊", "🎈", "🎁", "🎀", "🎯", "🎮", "🎲", "🎵", "🎶", "🎤", "🎧",
        "🎷", "🎸", "🎻", "🥁", "🏀", "⚽", "🏈", "🎾", "🥎", "🏐", "🏉", "🎳",
        "🎿", "🏂", "🏋", "🚴", "🚶", "🏌", "🤸"
    ]
}

# Hàm chọn biểu tượng dựa trên nhãn
def get_icons_by_rating(rating, num_icons=3):
    if rating in icons_mapping:
        icons = icons_mapping[rating]
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
            icon_position = random.choice(['middle', 'end'])  # Vị trí thêm biểu tượng
            if icon_position == 'middle':
                words = sentence.split()
                if len(words) > 1:
                    mid_index = len(words) // 2
                    words.insert(mid_index, icons[i])
                    sentences[i] = " ".join(words)
            elif icon_position == 'end':
                sentences[i] += f" {icons[i]}"
    return " ".join(sentences)

# Đọc dữ liệu từ file Excel
input_file = "res/test.csv"  # Đường dẫn file của bạn
data = pd.read_csv(input_file)

# Kiểm tra cột 'text' và 'rating' có tồn tại
if 'content' not in data.columns or 'rating' not in data.columns:
    raise KeyError("Cột 'content' hoặc 'rating' không tồn tại trong file dữ liệu!")

# Áp dụng hàm cho từng dòng dữ liệu
data['content'] = data.apply(
    lambda row: distribute_icons(row['content'], get_icons_by_rating(row['rating'], num_icons=3)), axis=1
)

# Lưu kết quả ra file Excel
output_file = 'res/test_emoji.csv'
data.to_csv(output_file, index=False)

print(f"Kết quả đã được lưu vào file: {output_file}")

