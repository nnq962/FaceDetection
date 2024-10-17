import cv2
import os
import time
import numpy as np
from deepface import DeepFace
from sklearn.metrics.pairwise import cosine_similarity
import csv

# Tải mô hình phát hiện khuôn mặt Haar Cascade
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Đường dẫn đến file metadata
metadata_path = 'metadata.csv'

# Đọc metadata CSV vào danh sách
metadata = []
with open(metadata_path, newline='') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        metadata.append(row)

# Mở webcam
cap = cv2.VideoCapture(0)  # Thay đổi thành 1 để sử dụng webcam thứ hai nếu cần

if not cap.isOpened():
    print("Không thể mở webcam.")
    exit()

frame_skip = 5  # Bỏ qua 5 khung hình để giảm tải xử lý
frame_count = 0
similarity_threshold = 0.6  # Ngưỡng xác định trùng khớp
last_embedding_time = 0  # Để kiểm tra thời gian giữa các lần lấy embedding
embedding_interval = 3  # Chỉ lấy embedding mỗi 3 giây
last_face_position = None  # Lưu trữ vị trí khuôn mặt trước đó để kiểm tra chuyển động

def get_embedding(frame):
    """Trích xuất embedding từ khung hình hiện tại"""
    try:
        embedding = DeepFace.represent(img_path=frame, model_name='Facenet')[0]["embedding"]
        return embedding
    except Exception as e:
        print(f"Lỗi khi trích xuất embedding: {str(e)}")
        return None

def compare_embedding(test_embedding, metadata, similarity_threshold):
    """So sánh embedding trích xuất với các embedding đã lưu trong metadata"""
    for row in metadata:
        embedding_path = row['Path']
        with open(embedding_path, 'r') as f:
            stored_embedding = json.load(f)[0]["embedding"]
            similarity = cosine_similarity([test_embedding], [stored_embedding])[0][0]
            if similarity > similarity_threshold:
                return f"Ảnh trùng khớp với {row['Name']} (ID: {row['ID']}). Độ tương đồng: {similarity:.4f}"
    return "Không tìm thấy data về ảnh này."

while True:
    ret, frame = cap.read()
    if not ret:
        print("Không thể nhận diện khung hình.")
        break

    frame_count += 1
    if frame_count % frame_skip != 0:
        continue  # Bỏ qua nếu không đạt số khung hình cần thiết để xử lý

    # Lật khung hình theo chiều ngang
    flipped_frame = cv2.flip(frame, 1)

    # Chuyển đổi khung hình sang ảnh xám để phát hiện khuôn mặt
    gray = cv2.cvtColor(flipped_frame, cv2.COLOR_BGR2GRAY)

    # Phát hiện khuôn mặt
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    if len(faces) > 0:
        # Lấy khuôn mặt đầu tiên
        (x, y, w, h) = faces[0]
        face_position = (x, y, w, h)

        # Vẽ hình chữ nhật quanh khuôn mặt phát hiện được
        cv2.rectangle(flipped_frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

        # Kiểm tra thời gian giữa các lần trích xuất embedding
        current_time = time.time()
        if current_time - last_embedding_time > embedding_interval:
            # Kiểm tra xem khuôn mặt có chuyển động không (dựa trên vị trí)
            if last_face_position is None or np.linalg.norm(np.array(face_position) - np.array(last_face_position)) > 20:
                # Trích xuất embedding từ khung hình có khuôn mặt
                face_roi = flipped_frame[y:y+h, x:x+w]
                test_embedding = get_embedding(face_roi)

                if test_embedding is not None:
                    # So sánh với metadata
                    result = compare_embedding(test_embedding, metadata, similarity_threshold)
                    print(result)

                last_embedding_time = current_time
                last_face_position = face_position

    # Hiển thị khung hình đã lật
    cv2.imshow('Webcam - Flipped', flipped_frame)

    # Nhấn 'q' để thoát
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Giải phóng webcam và đóng tất cả các cửa sổ
cap.release()
cv2.destroyAllWindows()
