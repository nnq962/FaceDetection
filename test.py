import faiss
import numpy as np
import os
import json
import cv2
import insightface

# Hàm để chuẩn hóa embeddings (cosine similarity)
def normalize_vector(vectors):
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors / norms

# Khởi tạo mô hình InsightFace
model = insightface.app.FaceAnalysis(providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
model.prepare(ctx_id=0)

# Đường dẫn tới thư mục chứa các tệp JSON embedding
embedding_dir = 'embeddings'

# Tạo FAISS index (sử dụng L2 cho cosine distance bằng cách chuẩn hóa vectors)
dimension = 512  # Độ dài vector embedding
index = faiss.IndexFlatL2(dimension)  # Sử dụng L2 distance

# Danh sách để lưu tên của từng embedding
embedding_names = []

# Tải các embeddings từ JSON và thêm vào FAISS index
for json_file in os.listdir(embedding_dir):
    if json_file.endswith('.json'):
        file_path = os.path.join(embedding_dir, json_file)

        with open(file_path, 'r') as f:
            data = json.load(f)
            person_name = data['name']
            embeddings_list = data['embeddings']

            for embedding_data in embeddings_list:
                embedding = np.array(embedding_data['embedding'], dtype=np.float32)
                
                # Chuẩn hóa embedding trước khi thêm vào FAISS (để mô phỏng cosine similarity)
                normalized_embedding = normalize_vector(np.expand_dims(embedding, axis=0))

                # Thêm embedding vào FAISS index
                index.add(normalized_embedding)
                embedding_names.append(person_name)

# Hàm để tìm kiếm khuôn mặt gần nhất với FAISS
def search_with_faiss(image_path):
    # Trích xuất embedding từ ảnh cần tìm kiếm
    image = cv2.imread(image_path)
    faces = model.get(image)

    if len(faces) > 0:
        query_embedding = faces[0].embedding.astype(np.float32)
        query_embedding = np.expand_dims(query_embedding, axis=0)
        
        # Chuẩn hóa query embedding để tính toán cosine similarity
        normalized_query = normalize_vector(query_embedding)

        # Tìm k kết quả gần nhất (ví dụ, k=1)
        k = 1
        distances, indices = index.search(normalized_query, k)

        # Trả về tên của kết quả gần nhất
        closest_index = indices[0][0]
        closest_name = embedding_names[closest_index]
        closest_distance = distances[0][0]

        print(f"Hình ảnh được xác định là: {closest_name} với khoảng cách {closest_distance}")
    else:
        print("Không phát hiện được khuôn mặt trong ảnh mới.")

# Đường dẫn tới ảnh cần tìm kiếm
image_path = 'guma.jpg'
search_with_faiss(image_path)
