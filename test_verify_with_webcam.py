import faiss
import numpy as np
import cv2
import os
import insightface
import json
from sklearn.metrics.pairwise import cosine_similarity

# Hàm tính cosine similarity
def calculate_cosine_similarity(vector1, vector2):
    vector1 = np.expand_dims(vector1, axis=0)
    vector2 = np.expand_dims(vector2, axis=0)
    return cosine_similarity(vector1, vector2)[0][0]

# Khởi tạo mô hình InsightFace
model = insightface.app.FaceAnalysis(providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
model.prepare(ctx_id=0)

# Đường dẫn tới file Haar Cascade cho phát hiện khuôn mặt
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')


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

# Hàm tìm kiếm với FAISS và so sánh bằng cosine similarity
def search_with_faiss_and_cosine(query_embedding, threshold=0.7):
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
            return embedding_names[idx], similarity  # Trả về tên và độ tương đồng
    return None, None

# Khởi tạo webcam
cap = cv2.VideoCapture(0)

while True:
    # Đọc frame từ webcam
    ret, frame = cap.read()
    if not ret:
        print("Không thể truy cập webcam.")
        break

    # Chuyển sang thang độ xám cho phát hiện khuôn mặt
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Phát hiện khuôn mặt
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    if len(faces) > 0:
        # Lấy khuôn mặt đầu tiên
        (x, y, w, h) = faces[0]
        face_position = (x, y, w, h)

        # Vẽ hình chữ nhật quanh khuôn mặt phát hiện được
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)


    for (x, y, w, h) in faces:
        # Vẽ hình chữ nhật quanh khuôn mặt
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

        # Cắt khuôn mặt từ frame
        face_img = frame[y:y+h, x:x+w]

        # Trích xuất embedding từ khuôn mặt
        faces_in_frame = model.get(face_img)

        print('------------------')

        if len(faces_in_frame) > 0:
            face_embedding = faces_in_frame[0].embedding.astype(np.float32)

            # So sánh với database
            name, similarity = search_with_faiss_and_cosine(face_embedding, threshold=0.7)
            
            if name:
                cv2.putText(frame, f"Name: {name}, Similarity: {similarity:.2f}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)
            else:
                cv2.putText(frame, "Unknown", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
    
    # Hiển thị frame
    cv2.imshow('Webcam Face Recognition', frame)

    # Thoát khi nhấn phím 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Giải phóng webcam và đóng các cửa sổ
cap.release()
cv2.destroyAllWindows()
