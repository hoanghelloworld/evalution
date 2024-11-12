import pandas as pd
import re

def processing(file_path):
    # Đọc file CSV
    result_df = pd.read_csv(file_path)

    # Hàm để trích xuất văn bản và điểm số
    def extract_text_score(row):
        text_match = re.search(r"page_content='(.+?)'", row)
        score_match = re.search(r", ([\d.]+)\)$", row)
        text = text_match.group(1) if text_match else None
        score = float(score_match.group(1)) if score_match else None
        return pd.Series([text, score])

    # Áp dụng hàm trích xuất vào cột 'Relevant Documents'
    result_df[['Văn bản tham chiếu', 'Score']] = result_df['Relevant Documents'].apply(extract_text_score)
    # Trả về DataFrame đã được xử lý
    return result_df
