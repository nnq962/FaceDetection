import faiss
import numpy as np
import cv2
import os
import json
import insightface
from sklearn.metrics.pairwise import cosine_similarity

# Hàm tính cosine similarity
def calculate_cosine_similarity(vector1, vector2):
    vector1 = np.expand_dims(vector1, axis=0)
    vector2 = np.expand_dims(vector2, axis=0)
    return cosine_similarity(vector1, vector2)[0][0]

# Khởi tạo mô hình InsightFace
model = insightface.app.FaceAnalysis(providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
model.prepare(ctx_id=0)

# Đường dẫn tới thư mục chứa embeddings và FAISS index
embedding_dir = 'embeddings'

# Tạo FAISS index (sử dụng L2)
dimension = 512  # Độ dài vector embedding
index = faiss.IndexFlatL2(dimension)

# Danh sách để lưu tên của từng embedding
embedding_names = []
embeddings = []

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
                
                # Thêm embedding vào FAISS index
                index.add(np.expand_dims(embedding, axis=0))
                embedding_names.append(person_name)
                embeddings.append(embedding)

# Hàm tìm kiếm với FAISS và sau đó so sánh kỹ hơn bằng cosine similarity
def search_with_faiss_and_cosine(image_path, threshold=0.7):
    # Trích xuất embedding từ ảnh cần tìm kiếm
    image = cv2.imread(image_path)
    faces = model.get(image)

    if len(faces) > 0:
        query_embedding = faces[0].embedding.astype(np.float32)
        query_embedding = np.expand_dims(query_embedding, axis=0)

        # Tìm top-k (ví dụ top-5) kết quả gần nhất từ FAISS
        k = 5
        distances, indices = index.search(query_embedding, k)

        match_found = False

        # So sánh với cosine similarity và áp dụng ngưỡng
        for idx in indices[0]:
            candidate_embedding = embeddings[idx]
            similarity = calculate_cosine_similarity(query_embedding[0], candidate_embedding)

            if similarity >= threshold:
                match_found = True
                print(f"Hình ảnh được xác định là: {embedding_names[idx]} với cosine similarity {similarity}")
                break  # Dừng lại khi đã tìm thấy người có cosine similarity >= threshold

        if not match_found:
            print("Không tìm thấy người nào trong database với độ tương đồng đủ lớn.")
    else:
        print("Không phát hiện được khuôn mặt trong ảnh mới.")

# Đường dẫn tới ảnh cần tìm kiếm
image_path = 'faker.jpg'
# Gọi hàm với ngưỡng cosine similarity = 0.7
search_with_faiss_and_cosine(image_path, threshold=0.7)
