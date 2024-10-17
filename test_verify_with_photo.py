import os
import json
import pandas as pd
from deepface import DeepFace
from sklearn.metrics.pairwise import cosine_similarity

# Đường dẫn đến thư mục chứa các embedding đã lưu
embedding_dir = 'embeddings'
metadata_file = 'metadata.csv'

# 1. Trích xuất embedding từ ảnh bạn muốn so sánh
test_image_path = 'faker.jpg'
test_embedding = DeepFace.represent(img_path=test_image_path, model_name='Facenet')[0]["embedding"]

# 2. Đọc metadata từ file CSV
metadata = pd.read_csv(metadata_file)

# Ngưỡng xác định giống nhau (tùy chỉnh)
similarity_threshold = 0.6

# Biến cờ để kiểm tra xem có tìm thấy sự trùng khớp không
found_match = False

# 3. Duyệt qua các hàng trong metadata để so sánh
for index, row in metadata.iterrows():
    embedding_path = row['Path']  # Lấy đường dẫn từ cột 'Path'
    
    # Nạp embedding từ file JSON
    with open(embedding_path, 'r') as f:
        stored_embedding = json.load(f)[0]["embedding"]

    # 4. So sánh embedding từ ảnh với embedding đã lưu
    similarity = cosine_similarity([test_embedding], [stored_embedding])[0][0]  # Tính cosine similarity
    
    # 5. Đánh giá mức độ giống nhau
    if similarity > similarity_threshold:
        print("===========================================================")
        print(f"Ảnh trùng khớp với thông tin của {row['Name']} (ID: {row['ID']}). Độ tương đồng: {similarity}")
        found_match = True  # Đánh dấu là đã tìm thấy sự trùng khớp
        break  # Dừng lại nếu đã tìm thấy sự trùng khớp

# Nếu không có sự trùng khớp nào
if not found_match:
    print("===========================================================")
    print("Không tìm thấy data về ảnh này.")
